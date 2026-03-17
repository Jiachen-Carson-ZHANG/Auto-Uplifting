from __future__ import annotations
import os
import uuid
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any

from src.models.task import TaskSpec, ExperimentPlan, RunConfig
from src.models.results import DataProfile, RunEntry, RunResult, RunDiagnostics
from src.models.nodes import ExperimentNode, NodeStage, NodeStatus, SearchContext
from src.llm.backend import LLMBackend
from src.memory.run_store import RunStore
from src.orchestration.state import ExperimentTree
from src.orchestration.scheduler import Scheduler
from src.orchestration.accept_reject import AcceptReject
from src.agents.manager import ExperimentManager, ActionType
from src.agents.selector import SelectorAgent
from src.execution.config_mapper import ConfigMapper
from src.execution.autogluon_runner import AutoGluonRunner
from src.execution.result_parser import ResultParser


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
    ) -> None:
        self.task = task
        self._llm = llm
        self._higher_is_better = higher_is_better

        # Session output directory
        session_name = f"{datetime.now().strftime('%Y-%m-%d')}_{task.task_name}"
        self._session_dir = Path(experiments_dir) / session_name
        self._session_dir.mkdir(parents=True, exist_ok=True)

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
        self._runner = AutoGluonRunner(target_column=task.target_column)
        self._seed_ideas = seed_ideas or []
        self._run_counter = 0

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

        # Update node status to running
        self.tree.update_node(node.model_copy(update={
            "status": NodeStatus.RUNNING,
            "config": config,
        }))

        result = self._runner.run(config)

        diagnostics = RunDiagnostics(
            data_profile_ref=str(self._session_dir / "data_profile.json"),
            failure_mode="execution_error" if result.status == "failed" else None,
        )

        entry = RunEntry(
            run_id=run_id,
            node_id=node.node_id,
            config=config,
            result=result,
            diagnostics=diagnostics,
            agent_rationale=node.plan.rationale,
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
        print(f"\n{'='*60}")
        print(f"Session: {self.task.task_name}")
        print(f"{'='*60}\n")

        # Step 1: Profile data
        data_profile = self.profile_data()
        print(f"Data profile: {data_profile.summary}")

        # Step 2: Load hypotheses
        hyps = hypotheses or self._seed_ideas
        if not hyps:
            hyps = [{"hypothesis": "Try GBM baseline with default settings", "rationale": "Safe default"}]

        # Step 3: Create candidate root nodes
        print(f"\nCreating {len(hyps)} candidate nodes...")
        candidate_nodes = self.create_candidate_nodes(hyps, data_profile)

        # Step 4: Warm-up loop
        print(f"\n--- WARM-UP PHASE ({len(candidate_nodes)} candidates) ---")
        for node in candidate_nodes:
            print(f"  Running candidate: {node.node_id} | metric={node.plan.eval_metric}")
            entry = self.execute_node(node, data_profile)
            fresh_node = self.tree.get_node(node.node_id)
            self.scheduler.record_warmup_run(fresh_node)
            metric = entry.result.primary_metric
            status = entry.result.status
            print(f"  → {status} | primary_metric={metric}")

        if self.scheduler.should_advance_to_optimization():
            self.scheduler.advance_to_optimization()

        # Step 5: Optimization loop
        incumbent = self.tree.get_incumbent(higher_is_better=self._higher_is_better)
        if incumbent is None:
            print("\nNo valid incumbent after warm-up. Stopping.")
            return None

        print(f"\n--- OPTIMIZE PHASE | incumbent={incumbent.node_id} "
              f"metric={incumbent.primary_metric():.4f} ---")

        while not self.scheduler.should_stop():
            context = SearchContext(
                task=self.task,
                data_profile=data_profile,
                history=self.run_store.get_history(),
                incumbent=self.run_store.get_incumbent(self._higher_is_better),
                current_node=incumbent,
                stage="optimize",
                budget_remaining=self.scheduler.max_optimize_iterations - self.scheduler._optimize_count,
                budget_used=self.scheduler._optimize_count,
                similar_cases=[],
                failed_attempts=self.run_store.get_failed(),
            )

            action = self._manager.next_action(context)
            if action.action_type == ActionType.STOP:
                print(f"  Manager says STOP: {action.reason}")
                break

            # Use selector to propose refinement (refiner agent added in Phase 3)
            plan = self._selector.select(
                hypothesis=(
                    f"Refine the current best config (metric={incumbent.primary_metric():.4f}). "
                    f"Try ONE improvement: consider changing validation strategy, "
                    f"model families, or increasing time budget."
                ),
                task=self.task,
                data_profile=data_profile,
                prior_runs=self.run_store.get_history(),
            )

            child_node = self.tree.add_child(
                parent_id=incumbent.node_id,
                plan=plan,
                edge_label=f"refinement attempt {self.scheduler._optimize_count + 1}: {plan.rationale[:80]}",
                stage=NodeStage.OPTIMIZE,
            )

            print(f"  Optimize run {self.scheduler._optimize_count + 1}: {child_node.node_id}")
            entry = self.execute_node(child_node, data_profile)
            fresh_child = self.tree.get_node(child_node.node_id)
            self.scheduler.record_optimize_run()

            accepted = self._accept_reject.evaluate(incumbent, fresh_child)
            if accepted:
                incumbent = fresh_child
                print(f"  → ACCEPTED | metric={fresh_child.primary_metric()}")
            else:
                print(f"  → REJECTED | metric={fresh_child.primary_metric()} "
                      f"(no improvement over {incumbent.primary_metric()})")

        # Save tree
        self.tree.save(self._session_dir / "tree.json")

        best_entry = self.run_store.get_incumbent(self._higher_is_better)
        if best_entry:
            print(f"\n{'='*60}")
            print(f"Best result: {best_entry.result.primary_metric:.4f} "
                  f"({self.task.eval_metric})")
            print(f"Best model: {best_entry.result.best_model_name}")
            print(f"{'='*60}\n")
        return best_entry
