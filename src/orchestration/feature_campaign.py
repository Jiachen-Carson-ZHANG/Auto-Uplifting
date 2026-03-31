"""
Feature engineering campaign orchestrator — sibling to CampaignOrchestrator.

Runs a baseline session, then iterates feature engineering proposals:
  1. Baseline ExperimentSession (identity preprocessing)
  2. Collect DataProfile, leaderboard, feature importances
  3. Feature iteration loop:
     a. FeatureEngineeringAgent.propose_and_execute()
     b. If success: save feature-engineered CSV, retrain
     c. Store FeatureHistoryEntry
     d. Check stop conditions
  4. Return campaign result

Uses empirical experiment memory (FeatureHistoryStore) + static reference packs.
No vector RAG, no CaseStore, no PreprocessingStore, no EmbeddingRetriever.
"""
from __future__ import annotations
import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Literal, Optional

import pandas as pd

from src.agents.feature_engineer import FeatureEngineeringAgent
from src.features.history import FeatureHistoryStore
from src.features.registry import build_default_registry
from src.llm.backend import LLMBackend
from src.models.campaign import (
    CampaignResult,
    FeatureCampaignConfig,
    SessionSummary,
)
from src.models.feature_engineering import FeatureHistoryEntry
from src.models.task import TaskSpec
from src.session import ExperimentSession


class FeatureCampaignOrchestrator:
    """
    Outer loop: baseline session → feature engineering iterations → retrain.

    Stops on: budget exhausted, plateau, consecutive blocks,
    or consecutive codegen failures.
    """

    def __init__(
        self,
        task: TaskSpec,
        llm: LLMBackend,
        config: Optional[FeatureCampaignConfig] = None,
        experiments_dir: str = "experiments",
        num_candidates: int = 3,
        max_optimize_iterations: int = 5,
        higher_is_better: bool = True,
    ) -> None:
        self._task = task
        self._llm = llm
        self._config = config or FeatureCampaignConfig()
        self._experiments_dir = experiments_dir
        self._num_candidates = num_candidates
        self._max_optimize_iterations = max_optimize_iterations
        self._higher_is_better = higher_is_better

        # Feature engineering components — no RAG
        registry = build_default_registry()
        self._fe_agent = FeatureEngineeringAgent(llm=llm, registry=registry)
        self._history_store = FeatureHistoryStore(self._config.feature_history_path)

        self._log = logging.getLogger(f"feature_campaign.{task.task_name}")
        self._log.setLevel(logging.DEBUG)
        if not self._log.handlers:
            fmt = logging.Formatter(
                "%(asctime)s %(levelname)s: %(message)s", datefmt="%H:%M:%S"
            )
            sh = logging.StreamHandler()
            sh.setFormatter(fmt)
            self._log.addHandler(sh)

    def run(self) -> CampaignResult:
        campaign_id = str(uuid.uuid4())[:8]
        started_at = datetime.now().isoformat()
        campaign_dir = (
            Path(self._experiments_dir)
            / "campaigns"
            / f"{campaign_id}_{self._task.task_name}_features"
        )
        sessions_dir = campaign_dir / "sessions"
        sessions_dir.mkdir(parents=True, exist_ok=True)

        # File handler
        if not any(isinstance(h, logging.FileHandler) for h in self._log.handlers):
            fh = logging.FileHandler(campaign_dir / "campaign.log", mode="a")
            fh.setFormatter(
                logging.Formatter(
                    "%(asctime)s %(levelname)s: %(message)s", datefmt="%H:%M:%S"
                )
            )
            self._log.addHandler(fh)

        self._log.info("=" * 60)
        self._log.info(
            f"Feature Campaign {campaign_id} | task={self._task.task_name} | "
            f"max_iterations={self._config.max_feature_iterations}"
        )
        self._log.info("=" * 60)

        sessions: List[SessionSummary] = []
        metrics: List[float] = []
        consecutive_blocks = 0
        consecutive_codegen_failures = 0
        total_sessions = 0  # tracks all ExperimentSessions (baseline + retrains)

        # ── Step 1: Baseline session (identity preprocessing) ───────
        self._log.info("--- Baseline Session ---")
        baseline_metric, baseline_session = self._run_session(
            sessions_dir, data_path=self._task.data_path
        )
        sessions.append(
            self._make_summary(baseline_session, baseline_metric, "identity")
        )
        total_sessions += 1
        if baseline_metric is not None:
            metrics.append(baseline_metric)
            self._log.info(f"Baseline metric: {baseline_metric:.4f}")
        else:
            self._log.warning("Baseline session had no successful runs.")

        # Save partial result
        self._save(
            self._build_result(campaign_id, started_at, sessions, "budget"),
            campaign_dir,
        )

        # ── Step 2: Collect profile for agent context ───────────────
        data_profile = self._get_data_profile()
        leaderboard = self._get_leaderboard(baseline_session)
        feature_importances = self._get_feature_importances(baseline_session)

        # ── Step 3: Feature iteration loop ──────────────────────────
        current_data_path = self._task.data_path

        for i in range(self._config.max_feature_iterations):
            self._log.info(
                f"--- Feature Iteration {i + 1}/{self._config.max_feature_iterations} ---"
            )

            df = pd.read_csv(current_data_path)
            incumbent_metric = metrics[-1] if metrics else None

            decision, audit, result, result_df = self._fe_agent.propose_and_execute(
                task=self._task,
                data_profile=data_profile,
                df=df,
                leaderboard=leaderboard,
                feature_importances=feature_importances,
                history=self._history_store.get_history(),
                incumbent_metric=incumbent_metric,
                budget_remaining=self._config.max_feature_iterations - i - 1,
                budget_used=i + 1,
            )

            self._log.info(
                f"Decision: action={decision.action} status={decision.status} "
                f"audit={audit.verdict} exec={result.status}"
            )

            if result.status in ("blocked", "failed"):
                consecutive_blocks += 1
                if decision.action == "escalate_codegen":
                    consecutive_codegen_failures += 1

                # Record history entry
                self._store_history(
                    decision, result, incumbent_metric, None
                )

                # Check stop conditions
                if consecutive_blocks >= self._config.max_consecutive_blocks:
                    self._log.info(
                        f"Stopping: {consecutive_blocks} consecutive blocks."
                    )
                    break
                if (
                    consecutive_codegen_failures
                    >= self._config.max_consecutive_codegen_failures
                ):
                    self._log.info(
                        f"Stopping: {consecutive_codegen_failures} consecutive "
                        f"codegen failures."
                    )
                    break
                continue

            # Success path — retrain with feature-engineered data
            consecutive_blocks = 0
            consecutive_codegen_failures = 0

            # Save feature-engineered CSV (result_df returned by agent)
            fe_dir = campaign_dir / f"features_{i + 1}"
            fe_dir.mkdir(parents=True, exist_ok=True)
            fe_path = str(fe_dir / "featured_data.csv")

            if result_df is not None:
                result_df.to_csv(fe_path, index=False)
                current_data_path = fe_path
            else:
                self._log.warning("Success status but no result_df — reusing previous data.")
                df.to_csv(fe_path, index=False)
                current_data_path = fe_path

            # Check max_sessions before retraining
            if total_sessions >= self._config.max_sessions:
                self._log.info(
                    f"Stopping: reached max_sessions={self._config.max_sessions}."
                )
                break

            # Retrain
            self._log.info("Retraining with featured data...")
            new_metric, new_session = self._run_session(
                sessions_dir, data_path=current_data_path
            )
            total_sessions += 1
            sessions.append(
                self._make_summary(new_session, new_metric, "featured")
            )

            if new_metric is not None:
                metrics.append(new_metric)
                self._log.info(f"New metric: {new_metric:.4f}")
            else:
                self._log.warning("Retrain session had no successful runs.")

            # Store history
            self._store_history(decision, result, incumbent_metric, new_metric)

            # Update context for next iteration
            leaderboard = self._get_leaderboard(new_session)
            feature_importances = self._get_feature_importances(new_session)

            # Save partial
            self._save(
                self._build_result(campaign_id, started_at, sessions, "budget"),
                campaign_dir,
            )

            # Check plateau
            if self._is_plateau(metrics):
                self._log.info("Plateau detected — stopping.")
                result_obj = self._build_result(
                    campaign_id, started_at, sessions, "plateau"
                )
                self._save(result_obj, campaign_dir)
                return result_obj

        # Budget exhausted
        result_obj = self._build_result(campaign_id, started_at, sessions, "budget")
        self._save(result_obj, campaign_dir)
        self._log.info(
            f"Campaign complete: best={result_obj.best_metric} "
            f"stopped={result_obj.stopped_reason}"
        )
        return result_obj

    # ── Helpers ─────────────────────────────────────────────────────

    def _run_session(
        self, sessions_dir: Path, data_path: str
    ) -> tuple[Optional[float], Optional[ExperimentSession]]:
        """Run one ExperimentSession and return (best_metric, session)."""
        try:
            session = ExperimentSession(
                task=self._task,
                llm=self._llm,
                experiments_dir=str(sessions_dir),
                num_candidates=self._num_candidates,
                max_optimize_iterations=self._max_optimize_iterations,
                higher_is_better=self._higher_is_better,
                preprocessed_data_path=data_path,
            )
            session.run()
            incumbent = session.run_store.get_incumbent(
                higher_is_better=self._higher_is_better
            )
            metric = incumbent.result.primary_metric if incumbent else None
            return metric, session
        except Exception as exc:
            self._log.error(f"Session failed: {exc}")
            return None, None

    def _get_data_profile(self):
        """Build a DataProfile from the task data."""
        from src.models.results import DataProfile

        df = pd.read_csv(self._task.data_path, nrows=1000)
        numeric = int(df.select_dtypes(include="number").shape[1])
        categorical = int(df.shape[1] - numeric)
        return DataProfile(
            n_rows=len(df),
            n_features=df.shape[1],
            feature_types={"numeric": numeric, "categorical": categorical},
        )

    def _get_leaderboard(self, session: Optional[ExperimentSession]) -> List[dict]:
        """Extract leaderboard from session."""
        if session is None:
            return []
        try:
            incumbent = session.run_store.get_incumbent(
                higher_is_better=self._higher_is_better
            )
            if incumbent and incumbent.result.leaderboard:
                return [
                    {"model_name": m.model_name, "score_val": m.score_val}
                    for m in incumbent.result.leaderboard[:5]
                ]
        except Exception:
            pass
        return []

    def _get_feature_importances(
        self, session: Optional[ExperimentSession]
    ) -> Dict[str, float]:
        """Extract feature importances (empty dict if unavailable)."""
        # Feature importances require AutoGluon predictor access
        # which is not persisted — return empty for now
        return {}

    def _store_history(
        self,
        decision,
        result,
        metric_before: Optional[float],
        metric_after: Optional[float],
    ) -> None:
        """Persist a FeatureHistoryEntry."""
        spec_json = ""
        if decision.feature_spec is not None:
            spec_json = decision.feature_spec.model_dump_json()

        entry = FeatureHistoryEntry(
            entry_id=str(uuid.uuid4())[:8],
            action=decision.action,
            feature_spec_json=spec_json,
            dataset_name=self._task.task_name,
            task_type=self._task.task_type,
            metric_before=metric_before,
            metric_after=metric_after,
            observed_outcome=result.status,
            distilled_takeaway=decision.reasoning[:200] if decision.reasoning else "",
            audit_verdict="",
        )
        try:
            self._history_store.add(entry)
        except Exception as exc:
            self._log.warning(f"Failed to store history entry: {exc}")

    def _make_summary(
        self,
        session: Optional[ExperimentSession],
        metric: Optional[float],
        strategy: str,
    ) -> SessionSummary:
        return SessionSummary(
            session_id=(
                str(session._session_dir.name)
                if session is not None
                else f"failed_{uuid.uuid4().hex[:6]}"
            ),
            best_metric=metric,
            preprocessing_strategy=strategy,
            session_dir=(
                str(session._session_dir)
                if session is not None
                else ""
            ),
            duration_seconds=0.0,
        )

    def _is_plateau(self, metrics: List[float]) -> bool:
        if len(metrics) < self._config.plateau_window:
            return False
        recent = metrics[-self._config.plateau_window :]
        return max(recent) - min(recent) < self._config.plateau_threshold

    def _build_result(
        self,
        campaign_id: str,
        started_at: str,
        sessions: List[SessionSummary],
        stopped_reason: Literal["plateau", "budget"],
    ) -> CampaignResult:
        valid_metrics = [s.best_metric for s in sessions if s.best_metric is not None]
        best = (
            max(valid_metrics)
            if valid_metrics and self._higher_is_better
            else min(valid_metrics)
            if valid_metrics
            else None
        )
        best_sid = None
        if best is not None:
            for s in sessions:
                if s.best_metric == best:
                    best_sid = s.session_id
                    break
        return CampaignResult(
            campaign_id=campaign_id,
            task_name=self._task.task_name,
            started_at=started_at,
            sessions=sessions,
            best_metric=best,
            best_session_id=best_sid,
            stopped_reason=stopped_reason,
        )

    def _save(self, result: CampaignResult, campaign_dir: Path) -> None:
        path = campaign_dir / "campaign.json"
        path.write_text(result.model_dump_json(indent=2))
