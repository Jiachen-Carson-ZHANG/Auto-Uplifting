"""
Campaign-level data classes.

Origin  : defined by CampaignOrchestrator (src/orchestration/campaign.py)
Consumed: campaign.py entrypoint (display / persistence), future analysis tools
"""
from __future__ import annotations
from typing import List, Literal, Optional
from pydantic import BaseModel


class CampaignConfig(BaseModel):
    """
    Hyper-parameters for the outer campaign loop.
    All runs in a campaign share the same task and LLM.
    """
    max_sessions: int = 5
    plateau_threshold: float = 0.002   # stop if best metric moves < this across plateau_window sessions
    plateau_window: int = 3            # number of recent sessions to check for plateau
    preprocessing_bank_path: str = "experiments/preprocessing_bank.jsonl"  # Phase 4b


class SessionSummary(BaseModel):
    """
    One row in the campaign log — one per completed session.
    Written by CampaignOrchestrator after each session finishes.
    """
    session_id: str
    best_metric: Optional[float] = None            # None if all runs failed
    preprocessing_strategy: str = "identity"
    preprocessing_validation_passed: bool = False  # True if ValidationHarness accepted the code
    preprocessing_turns_used: int = 0              # ReAct turns consumed by PreprocessingAgent
    session_dir: str                               # absolute path to session artifacts
    duration_seconds: float
    error_message: Optional[str] = None            # set if the session raised an exception


class FeatureCampaignConfig(BaseModel):
    """
    Config for the feature engineering campaign loop.
    Uses empirical experiment memory + static reference packs — no vector RAG.
    """
    max_sessions: int = 10
    max_feature_iterations: int = 10
    plateau_threshold: float = 0.002
    plateau_window: int = 3
    feature_history_path: str = "experiments/feature_history.jsonl"
    max_consecutive_blocks: int = 3
    max_consecutive_codegen_failures: int = 2


class CampaignResult(BaseModel):
    """
    Full record of a completed campaign.
    Saved to experiments/campaigns/{campaign_id}/campaign.json.
    """
    campaign_id: str
    task_name: str
    started_at: str                        # ISO 8601
    sessions: List[SessionSummary]
    best_metric: Optional[float]           # best across all sessions
    best_session_id: Optional[str] = None
    stopped_reason: Literal['plateau', 'budget']
