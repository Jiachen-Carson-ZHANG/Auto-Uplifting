from __future__ import annotations
import json
import logging
import os
import uuid
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any

from src.models.task import TaskSpec, ExperimentPlan, RunConfig
from src.models.results import DataProfile, RunEntry, RunResult, RunDiagnostics
from src.models.nodes import ExperimentNode, NodeStage, NodeStatus, SearchContext, TaskTraits
from src.llm.backend import LLMBackend
from src.memory.run_store import RunStore
from src.orchestration.state import ExperimentTree
from src.orchestration.scheduler import Scheduler
from src.orchestration.accept_reject import AcceptReject
from src.agents.manager import ExperimentManager, ActionType
from src.agents.selector import SelectorAgent
from src.agents.refiner import RefinerAgent
from src.execution.config_mapper import ConfigMapper
from src.execution.autogluon_runner import AutoGluonRunner
from src.execution.result_parser import ResultParser
from src.memory.case_store import CaseStore
from src.memory.retrieval import CaseRetriever
from src.memory.distiller import Distiller
from src.memory.context_builder import ContextBuilder
from src.agents.ideator import IdeatorAgent
from src.memory.trait_utils import rows_bucket, features_bucket, balance_bucket


class ExperimentSession:
    """
    Wires all 4 layers together into a runnable experiment session.
    Owns the tree, run store, scheduler, and agent instances.
    """

    def __init__(
        self,
        task: TaskSpec,
        llm: LLMBackend,
        experiments_dir: str = "experiments",
        num_candidates: int = 3,
        max_optimize_iterations: int = 5,
        higher_is_better: bool = True,
        seed_ideas: Optional[List[Dict[str, str]]] = None,
        case_store_path: Optional[str] = None,
    ) -> None:
        self.task = task
        self._llm = llm
        self._higher_is_better = higher_is_better

        # Session output directory
        session_name = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{task.task_name}"
        self._session_dir = Path(experiments_dir) / session_name
        self._session_dir.mkdir(parents=True, exist_ok=True)

        # Session logger — writes to both stdout and session.log
        self._log = logging.getLogger(f"session.{task.task_name}")
        self._log.setLevel(logging.DEBUG)
        if not self._log.handlers:
            fmt = logging.Formatter("%(asctime)s %(levelname)s: %(message)s", datefmt="%H:%M:%S")
            sh = logging.StreamHandler()
            sh.setFormatter(fmt)
            fh = logging.FileHandler(self._session_dir / "session.log", mode="a")
            fh.setFormatter(fmt)
            self._log.addHandler(sh)
            self._log.addHandler(fh)

        # Core components
        self.run_store = RunStore(self._session_dir / "decisions.jsonl")
        self.tree = ExperimentTree()
        self.scheduler = Scheduler(
            num_candidates=num_candidates,
            max_optimize_iterations=max_optimize_iterations,
        )
        self._accept_reject = AcceptReject(higher_is_better=higher_is_better)
        self._manager = ExperimentManager(llm=llm)
        self._selector = SelectorAgent(llm=llm)
        self._refiner = RefinerAgent(llm=llm)
        self._runner = AutoGluonRunner(target_column=task.target_column)
        self._case_store = CaseStore(case_store_path) if case_store_path else None
        self._retriever = CaseRetriever()
        self._distiller = Distiller(llm=llm)
        self._context_builder = ContextBuilder()
        self._ideator = IdeatorAgent(llm=llm, num_hypotheses=num_candidates)
        self._run_counter = 0

        # Write session manifest — frozen snapshot of all controllable config
        manifest = {
            "session_name": session_name,
            "started_at": datetime.now().isoformat(),
            "task": task.model_dump(),
            "search": {
                "num_candidates": num_candidates,
                "max_optimize_iterations": max_optimize_iterations,
                "higher_is_better": higher_is_better,
            },
            "llm": {
                "provider": getattr(llm, "provider", type(llm).__name__),
                "model": getattr(llm, "model", "unknown"),
            },
            "case_store_path": case_store_path,
        }
        (self._session_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, default=str)
        )

    def profile_data(self) -> DataProfile:
        """Build a DataProfile from the task's dataset."""
        df = pd.read_csv(self.task.data_path)
        target = self.task.target_column
        feature_cols = [c for c in df.columns if c != target]
        n_features = len(feature_cols)

        numeric = sum(1 for c in feature_cols if pd.api.types.is_numeric_dtype(df[c]))
        categorical = n_features - numeric

        target_counts = df[target].value_counts()
        balance_ratio = float(target_counts.min() / target_counts.max()) if len(target_counts) > 1 else 1.0
        target_distribution = {str(k): int(v) for k, v in target_counts.items()}
        missing_rate = float(df.isnull().mean().mean())

        high_cardinality = [
            c for c in feature_cols
            if df[c].dtype == object and df[c].nunique() > 50
        ]

        return DataProfile(
            n_rows=len(df),
            n_features=n_features,
            feature_types={"numeric": numeric, "categorical": categorical},
            target_distribution=target_distribution,
            class_balance_ratio=balance_ratio,
            missing_rate=missing_rate,
            high_cardinality_cols=high_cardinality,
            suspected_leakage_cols=[],
            summary=(
                f"{len(df)} rows, {n_features} features, "
                f"class_balance={balance_ratio:.2f}, missing={missing_rate:.1%}"
            ),
        )

    def create_candidate_nodes(
        self,
        hypotheses: List[Dict[str, str]],
        data_profile: DataProfile,
    ) -> List[ExperimentNode]:
        """Convert hypotheses into root ExperimentNodes using the SelectorAgent."""
        nodes = []
        for h in hypotheses:
            plan = self._selector.select(
                hypothesis=h["hypothesis"],
                task=self.task,
                data_profile=data_profile,
                prior_runs=self.run_store.get_history(),
            )
            node = self.tree.add_root(plan=plan, stage=NodeStage.WARMUP)
            nodes.append(node)
        return nodes

    def execute_node(self, node: ExperimentNode, data_profile: DataProfile) -> RunEntry:
        """Run AutoGluon for a node and return the RunEntry."""
        self._run_counter += 1
        run_id = f"run_{self._run_counter:04d}"
        run_dir = str(self._session_dir / "runs" / run_id)

        config = ConfigMapper.to_run_config(
            plan=node.plan,
            run_id=run_id,
            node_id=node.node_id,
            data_path=self.task.data_path,
            output_dir=run_dir,
        )

        # Write per-run config snapshot — full autogluon kwargs + agent's ExperimentPlan
        Path(run_dir).mkdir(parents=True, exist_ok=True)
        run_config_snapshot = {
            "run_id": run_id,
            "node_id": node.node_id,
            "autogluon_kwargs": config.autogluon_kwargs,
            "experiment_plan": node.plan.model_dump(),
        }
        (Path(run_dir) / "run_config.json").write_text(
            json.dumps(run_config_snapshot, indent=2, default=str)
        )

        # Update node status to running
        self.tree.update_node(node.model_copy(update={
            "status": NodeStatus.RUNNING,
            "config": config,
        }))

        result = self._runner.run(config)

        diagnostics = RunDiagnostics(
            failure_mode="execution_error" if result.status == "failed" else None,
        )

        entry = RunEntry(
            run_id=run_id,
            node_id=node.node_id,
            config=config,
            result=result,
            diagnostics=diagnostics,
            agent_rationale=node.plan.rationale,
            plan=node.plan,
        )

        # Populate diagnostics from result and parent
        if result.status == "success":
            parent_node = self.tree.get_node(node.parent_id) if node.parent_id else None
            parent_metric = parent_node.primary_metric() if parent_node and parent_node.has_result() else None
            metric_vs_parent = None
            if parent_metric is not None and result.primary_metric is not None:
                metric_vs_parent = round(result.primary_metric - parent_metric, 4)
            entry.diagnostics = RunDiagnostics(
                overfitting_gap=result.diagnostics_overfitting_gap,
                metric_vs_parent=metric_vs_parent,
            )

        self.run_store.append(entry)

        new_status = NodeStatus.SUCCESS if result.status == "success" else NodeStatus.FAILED
        self.tree.update_node(node.model_copy(update={
            "status": new_status,
            "config": config,
            "entry": entry,
        }))

        return entry

    def run(self, hypotheses: Optional[List[Dict[str, str]]] = None) -> Optional[RunEntry]:
        """
        Full session loop:
        1. Profile data
        2. Create candidate nodes from hypotheses
        3. Warm-up: run each candidate
        4. Optimize: refine the incumbent
        5. Return best RunEntry
        """
        self._log.info("=" * 60)
        self._log.info("Session: %s", self.task.task_name)
        self._log.info("=" * 60)

        # Step 1: Profile data
        data_profile = self.profile_data()
        self._log.info("Data profile: %s", data_profile.summary)

        # Step 2: Load hypotheses
        # Retrieve similar past cases for grounding
        similar_cases = []
        if self._case_store:
            query_traits = TaskTraits(
                task_type=self.task.task_type,
                n_rows_bucket=rows_bucket(data_profile.n_rows),
                n_features_bucket=features_bucket(data_profile.n_features),
                class_balance=balance_bucket(data_profile.class_balance_ratio),
                feature_types=data_profile.feature_types,
            )
            similar_cases = self._retriever.rank(query_traits, self._case_store.get_all(), top_k=3)
            self._log.info("Retrieved %d similar past cases", len(similar_cases))

        hyps = hypotheses or self._ideator.ideate(
            task=self.task,
            data_profile=data_profile,
            similar_cases=similar_cases,
        )

        # Step 3: Create candidate root nodes
        self._log.info("Creating %d candidate nodes...", len(hyps))
        candidate_nodes = self.create_candidate_nodes(hyps, data_profile)

        # Step 4: Warm-up loop
        self._log.info("--- WARM-UP PHASE (%d candidates) ---", len(candidate_nodes))
        for node in candidate_nodes:
            self._log.info("Running candidate: %s | metric=%s", node.node_id, node.plan.eval_metric)
            entry = self.execute_node(node, data_profile)
            fresh_node = self.tree.get_node(node.node_id)
            self.scheduler.record_warmup_run(fresh_node)
            metric = entry.result.primary_metric
            status = entry.result.status
            self._log.info("  → %s | primary_metric=%s", status, metric)

        if self.scheduler.should_advance_to_optimization():
            self.scheduler.advance_to_optimization()

        # Step 5: Optimization loop
        incumbent = self.tree.get_incumbent(higher_is_better=self._higher_is_better)
        if incumbent is None:
            self._log.warning("No valid incumbent after warm-up. Stopping.")
            return None

        self._log.info(
            "--- OPTIMIZE PHASE | incumbent=%s metric=%.4f ---",
            incumbent.node_id, incumbent.primary_metric(),
        )

        while not self.scheduler.should_stop():
            context = self._context_builder.build(
                task=self.task,
                data_profile=data_profile,
                history=self.run_store.get_history(),
                incumbent=self.run_store.get_incumbent(self._higher_is_better),
                current_node=incumbent,
                stage="optimize",
                budget_remaining=self.scheduler.max_optimize_iterations - self.scheduler._optimize_count,
                budget_used=self.scheduler._optimize_count,
                similar_cases=similar_cases,
            )

            action = self._manager.next_action(context)
            if action.action_type == ActionType.STOP:
                self._log.info("Manager says STOP: %s", action.reason)
                break

            incumbent_entry = self.run_store.get_incumbent(self._higher_is_better)
            plan = self._refiner.refine(
                incumbent=incumbent_entry,
                task=self.task,
                prior_runs=self.run_store.get_history(),
            )

            child_node = self.tree.add_child(
                parent_id=incumbent.node_id,
                plan=plan,
                edge_label=f"refinement attempt {self.scheduler._optimize_count + 1}: {plan.rationale[:80]}",
                stage=NodeStage.OPTIMIZE,
            )

            self._log.info("Optimize run %d: %s", self.scheduler._optimize_count + 1, child_node.node_id)
            entry = self.execute_node(child_node, data_profile)
            fresh_child = self.tree.get_node(child_node.node_id)
            self.scheduler.record_optimize_run()

            accepted = self._accept_reject.evaluate(incumbent, fresh_child)
            if accepted:
                incumbent = fresh_child
                self._log.info("  → ACCEPTED | metric=%s", fresh_child.primary_metric())
            else:
                self._log.info(
                    "  → REJECTED | metric=%s (no improvement over %s)",
                    fresh_child.primary_metric(), incumbent.primary_metric(),
                )

        # Save tree
        self.tree.save(self._session_dir / "tree.json")

        if self._case_store:
            self._log.info("Distilling session into CaseStore...")
            try:
                case = self._distiller.distill(
                    task=self.task,
                    data_profile=data_profile,
                    run_history=self.run_store.get_history(),
                )
                self._case_store.add(case)
                self._log.info("Session distilled → case_id=%s", case.case_id)
            except Exception as e:
                self._log.warning("Distillation failed (non-fatal): %s", e)

        best_entry = self.run_store.get_incumbent(self._higher_is_better)
        if best_entry:
            self._log.info("=" * 60)
            self._log.info(
                "Best result: %.4f (%s) | model: %s",
                best_entry.result.primary_metric,
                self.task.eval_metric,
                best_entry.result.best_model_name,
            )
            self._log.info("=" * 60)
        return best_entry
