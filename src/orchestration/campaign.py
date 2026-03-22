"""
Outer optimization loop over multiple ExperimentSessions.

Origin  : campaign.py entrypoint (repo root)
Consumed: nothing downstream — writes campaign.json and campaign.log to disk

Each iteration:
  1. PreprocessingAgent generates a PreprocessingPlan (or identity fallback)
  2. PreprocessingExecutor applies the plan to the raw CSV
  3. ExperimentSession runs warm-up + optimize loop on the preprocessed data
  4. CampaignOrchestrator records SessionSummary, checks stop conditions

Sessions are stored inside campaigns/{campaign_id}/sessions/ so each campaign
is a self-contained folder.
"""
from __future__ import annotations
import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Literal, Optional

from src.models.campaign import CampaignConfig, SessionSummary, CampaignResult
from src.models.task import TaskSpec
from src.models.preprocessing import PreprocessingPlan, PreprocessingEntry
from src.llm.backend import LLMBackend
from src.execution.preprocessing_runner import PreprocessingExecutor
from src.memory.preprocessing_store import PreprocessingStore
from src.memory.embedding_retriever import EmbeddingRetriever
from src.agents.preprocessing_agent import PreprocessingAgent
from src.session import ExperimentSession


class CampaignOrchestrator:
    """
    Runs multiple ExperimentSessions on the same task, stopping when the
    metric plateaus or the session budget is exhausted.

    Sessions are stored in campaigns/{campaign_id}/sessions/ for easy navigation.
    campaign.json is written after each session so partial results survive crashes.

    Phase 4a: always uses identity preprocessing.
    Phase 4b: will generate new preprocessing strategies on plateau.
    """

    def __init__(
        self,
        task: TaskSpec,
        llm: LLMBackend,
        config: Optional[CampaignConfig] = None,
        experiments_dir: str = "experiments",
        num_candidates: int = 3,
        max_optimize_iterations: int = 5,
        higher_is_better: bool = True,
        case_store_path: Optional[str] = None,
    ) -> None:
        self._task = task
        self._llm = llm
        self._config = config or CampaignConfig()
        self._experiments_dir = experiments_dir
        self._num_candidates = num_candidates
        self._max_optimize_iterations = max_optimize_iterations
        self._higher_is_better = higher_is_better
        self._case_store_path = case_store_path
        self._executor = PreprocessingExecutor()
        self._preprocessing_store = PreprocessingStore(self._config.preprocessing_bank_path)
        self._prep_agent = PreprocessingAgent(llm=self._llm)
        # embed_backend: only OpenAIBackend exposes embed() — not on the LLMBackend ABC
        self._embed_backend = self._llm if hasattr(self._llm, "embed") else None
        self._log = logging.getLogger(f"campaign.{task.task_name}")
        self._log.setLevel(logging.DEBUG)
        if not self._log.handlers:
            # Note: campaign dir isn't known yet at __init__ time, so FileHandler is added in run()
            fmt = logging.Formatter("%(asctime)s %(levelname)s: %(message)s", datefmt="%H:%M:%S")
            sh = logging.StreamHandler()
            sh.setFormatter(fmt)
            self._log.addHandler(sh)

        # Auto-seed the preprocessing bank on first use (empty bank only)
        if not self._preprocessing_store.get_all():
            seeds_path = (
                Path(__file__).parent.parent.parent
                / "data" / "seeds" / "preprocessing_seeds.jsonl"
            )
            self._preprocessing_store.seed_from_file(str(seeds_path), self._embed_backend)

        # Warn if bank has entries with no embeddings and we can't embed them
        if self._embed_backend is None:
            null_count = sum(
                1 for e in self._preprocessing_store.get_all() if e.embedding is None
            )
            if null_count > 0:
                self._log.warning(
                    "PreprocessingStore: %d entries have no embedding (embed_backend not "
                    "available — set OPENAI_API_KEY to enable semantic retrieval). "
                    "Using naive ordering.",
                    null_count,
                )

    def run(self) -> CampaignResult:
        campaign_id = str(uuid.uuid4())[:8]
        started_at = datetime.now().isoformat()
        campaign_dir = Path(self._experiments_dir) / "campaigns" / f"{campaign_id}_{self._task.task_name}"
        sessions_dir = campaign_dir / "sessions"
        sessions_dir.mkdir(parents=True, exist_ok=True)

        # Add FileHandler once campaign_dir is known
        if not any(isinstance(h, logging.FileHandler) for h in self._log.handlers):
            fh = logging.FileHandler(campaign_dir / "campaign.log", mode="a")
            fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s", datefmt="%H:%M:%S"))
            self._log.addHandler(fh)

        self._log.info("=" * 60)
        self._log.info(f"Campaign {campaign_id} | task={self._task.task_name} | max_sessions={self._config.max_sessions}")
        self._log.info("=" * 60)

        sessions: List[SessionSummary] = []
        metrics: List[float] = []
        consecutive_prep_failures = 0

        for i in range(self._config.max_sessions):
            self._log.info(f"--- Session {i + 1}/{self._config.max_sessions} ---")
            t_start = datetime.now()
            summary: Optional[SessionSummary] = None
            session = None

            try:
                plan = self._preprocessing_plan(session_index=i)
                self._log.info(
                    f"Preprocessing: strategy={plan.strategy} "
                    f"validation_passed={plan.validation_passed} turns_used={plan.turns_used}"
                )

                # Failure tracking for consecutive preprocessing failures
                if plan.strategy == "generated" and not plan.validation_passed:
                    consecutive_prep_failures += 1
                    if consecutive_prep_failures >= 2:
                        self._log.warning(
                            "PreprocessingAgent: %d consecutive validation failures — "
                            "prompts or data may need attention",
                            consecutive_prep_failures,
                        )
                else:
                    consecutive_prep_failures = 0

                prep_dir = campaign_dir / f"preprocessing_{i + 1}"
                preprocessed_path = self._executor.run(self._task.data_path, plan, str(prep_dir))

                session = ExperimentSession(
                    task=self._task,
                    llm=self._llm,
                    experiments_dir=str(sessions_dir),
                    num_candidates=self._num_candidates,
                    max_optimize_iterations=self._max_optimize_iterations,
                    higher_is_better=self._higher_is_better,
                    case_store_path=self._case_store_path,
                    preprocessed_data_path=preprocessed_path,
                )
                session.run()

                duration = (datetime.now() - t_start).total_seconds()
                incumbent = session.run_store.get_incumbent(higher_is_better=self._higher_is_better)
                best_metric = incumbent.result.primary_metric if incumbent else None

                summary = SessionSummary(
                    session_id=str(session._session_dir.name),
                    best_metric=best_metric,
                    preprocessing_strategy=plan.strategy,
                    preprocessing_validation_passed=plan.validation_passed,
                    preprocessing_turns_used=plan.turns_used,
                    session_dir=str(session._session_dir),
                    duration_seconds=duration,
                )
                if best_metric is not None:
                    metrics.append(best_metric)
                    self._log.info(f"Session {i + 1} best: {best_metric:.4f}")
                else:
                    self._log.warning(f"Session {i + 1}: no successful runs")

                # Store successful generated preprocessing for future sessions
                if plan.strategy == "generated" and plan.validation_passed and plan.code:
                    self._store_preprocessing_entry(plan, best_metric or 0.0)

            except Exception as exc:
                duration = (datetime.now() - t_start).total_seconds()
                self._log.error(f"Session {i + 1} failed: {exc}")
                sid = str(session._session_dir.name) if session is not None else f"session_{i + 1}_failed"
                sdir = str(session._session_dir) if session is not None else ""
                summary = SessionSummary(
                    session_id=sid,
                    best_metric=None,
                    preprocessing_strategy="identity",
                    session_dir=sdir,
                    duration_seconds=duration,
                    error_message=str(exc),
                )

            sessions.append(summary)
            # Write campaign.json after every session so partial results survive
            partial = self._build_result(campaign_id, started_at, sessions, "budget")
            self._save(partial, campaign_dir)

            if self._is_plateau(metrics):
                self._log.info("Plateau detected — stopping campaign.")
                result = self._build_result(campaign_id, started_at, sessions, "plateau")
                self._save(result, campaign_dir)
                return result

        result = self._build_result(campaign_id, started_at, sessions, "budget")
        self._save(result, campaign_dir)
        self._log.info(f"Campaign complete: best={result.best_metric} | stopped={result.stopped_reason}")
        return result

    def _is_plateau(self, metrics: List[float]) -> bool:
        """True if the last plateau_window metrics are all within plateau_threshold of each other."""
        if len(metrics) < self._config.plateau_window:
            return False
        recent = metrics[-self._config.plateau_window:]
        return max(recent) - min(recent) < self._config.plateau_threshold

    def _best_metric(self, metrics: List[float]) -> Optional[float]:
        """Returns best metric respecting higher_is_better."""
        if not metrics:
            return None
        return max(metrics) if self._higher_is_better else min(metrics)

    def _preprocessing_plan(self, session_index: int = 0) -> PreprocessingPlan:
        """
        Call PreprocessingAgent to generate a preprocessing plan.
        Falls back to identity if the agent fails or returns strategy="identity".
        Session 0 always runs identity (warm-up baseline).
        """
        if session_index == 0:
            return PreprocessingPlan(strategy="identity")

        try:
            from src.models.results import DataProfile
            import pandas as pd
            df = pd.read_csv(self._task.data_path, nrows=1000)
            numeric = int(df.select_dtypes(include="number").shape[1])
            categorical = int(df.shape[1] - numeric)
            data_profile = DataProfile(
                n_rows=len(df),
                n_features=df.shape[1],
                feature_types={"numeric": numeric, "categorical": categorical},
            )
            candidates = self._preprocessing_store.get_similar(
                self._task.task_type, n=20
            )
            if self._embed_backend and candidates:
                query = (
                    f"{self._task.task_type} task on {self._task.task_name}: "
                    f"{self._task.description}"
                )
                similar_cases = EmbeddingRetriever(self._embed_backend).rank(
                    query, candidates, top_k=3
                )
            else:
                similar_cases = candidates[:3]
            plan = self._prep_agent.generate(
                task=self._task,
                data_profile=data_profile,
                data_path=self._task.data_path,
                similar_cases=similar_cases,
            )
            return plan
        except Exception as exc:
            self._log.warning("PreprocessingAgent call failed, using identity: %s", exc)
            return PreprocessingPlan(strategy="identity")

    def _store_preprocessing_entry(self, plan: PreprocessingPlan, metric_value: float) -> None:
        """Persist a validated preprocessing plan to the preprocessing bank."""
        entry = PreprocessingEntry(
            entry_id=str(uuid.uuid4())[:8],
            task_type=self._task.task_type,
            dataset_name=self._task.task_name,
            transformation_summary=plan.rationale or "generated by PreprocessingAgent",
            code=plan.code,
            metric_delta=metric_value,
        )
        if self._embed_backend:
            entry.embedding = self._embed_backend.embed(entry.transformation_summary)
            # embed() returns None on failure — stored as null, scores 0.0 in retrieval
        self._preprocessing_store.add(entry)
        self._log.info("PreprocessingStore: saved entry %s", entry.entry_id)

    def _build_result(
        self,
        campaign_id: str,
        started_at: str,
        sessions: List[SessionSummary],
        stopped_reason: Literal["plateau", "budget"],
    ) -> CampaignResult:
        metrics_with_values = [s.best_metric for s in sessions if s.best_metric is not None]
        best_metric = self._best_metric(metrics_with_values)
        best_session_id = None
        if best_metric is not None:
            best_session_id = max(
                (s for s in sessions if s.best_metric is not None),
                key=lambda s: s.best_metric if self._higher_is_better else -s.best_metric
            ).session_id
        return CampaignResult(
            campaign_id=campaign_id,
            task_name=self._task.task_name,
            started_at=started_at,
            sessions=sessions,
            best_metric=best_metric,
            best_session_id=best_session_id,
            stopped_reason=stopped_reason,
        )

    def _save(self, result: CampaignResult, campaign_dir: Path) -> None:
        path = campaign_dir / "campaign.json"
        path.write_text(result.model_dump_json(indent=2))
