import json

from cathode_screening.discovery.state import CampaignState


def test_campaign_state_defaults_for_legacy_payload(tmp_path):
    legacy_payload = {
        "campaign_name": "legacy_campaign",
        "created_at": "2026-03-03T00:00:00Z",
        "total_dft_queries": 0,
        "cycles": [],
        "pool_labeled_ids": [],
        "pool_remaining_ids": ["mp-1", "mp-2"],
        "current_cycle": 0,
        "current_stage": "screen",
        "current_model_artifacts": None,
    }
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(legacy_payload), encoding="utf-8")

    state = CampaignState.load(path)

    assert state.pool_source == "ensemble_soap_loco_test"
    assert state.ranking_strategy == "balanced"


def test_campaign_state_summary_includes_strategy_and_pool_source():
    state = CampaignState.new("demo", ["mp-1"])
    state.pool_source = "custom_pool"
    state.ranking_strategy = "explore"

    summary = state.summary()

    assert summary["pool_source"] == "custom_pool"
    assert summary["ranking_strategy"] == "explore"
