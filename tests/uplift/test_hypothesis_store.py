from src.models.uplift import UpliftHypothesis
from src.uplift.hypotheses import UpliftHypothesisStore, transition_hypothesis


def _hypothesis(action_type="window_sweep", status="proposed", trial_ids=None):
    return UpliftHypothesis(
        question="Does recency matter?",
        hypothesis_text="Recent purchase behavior should improve uplift.",
        stage_origin="diagnosis",
        action_type=action_type,
        status=status,
        trial_ids=trial_ids or [],
    )


def test_hypothesis_store_appends_and_loads_snapshots(tmp_path):
    store = UpliftHypothesisStore(tmp_path / "hypotheses.jsonl")
    hypothesis = _hypothesis()

    store.append(hypothesis)

    loaded = store.load_snapshots()
    assert len(loaded) == 1
    assert loaded[0].hypothesis_id == hypothesis.hypothesis_id


def test_hypothesis_store_latest_by_id_uses_last_snapshot(tmp_path):
    store = UpliftHypothesisStore(tmp_path / "hypotheses.jsonl")
    hypothesis = _hypothesis()
    store.append(hypothesis)
    updated = transition_hypothesis(hypothesis, "under_test", wave_id="WAVE-1")
    store.append(updated)

    latest = store.get_latest(hypothesis.hypothesis_id)

    assert latest is not None
    assert latest.status == "under_test"
    assert latest.wave_ids == ["WAVE-1"]


def test_hypothesis_store_queries_latest_records(tmp_path):
    store = UpliftHypothesisStore(tmp_path / "hypotheses.jsonl")
    h1 = _hypothesis(action_type="window_sweep", trial_ids=["RUN-1"])
    h2 = _hypothesis(action_type="cost_sensitivity", trial_ids=["RUN-2"])
    store.append(transition_hypothesis(h1, "under_test"))
    store.append(h2)

    assert [h.hypothesis_id for h in store.query_by_status("under_test")] == [
        h1.hypothesis_id
    ]
    assert [h.hypothesis_id for h in store.query_by_action_type("cost_sensitivity")] == [
        h2.hypothesis_id
    ]
    assert store.query_by_trial_id("RUN-1")[0].hypothesis_id == h1.hypothesis_id
