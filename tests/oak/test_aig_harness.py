from app.oak.eval import AIGHarness, AttackCase, default_attack_matrix
from app.oak.security import SinkOutcome


def test_bounded_carrier_matrix_routes_taint_and_blocks_every_unapproved_sink() -> None:
    results = AIGHarness().run_matrix(default_attack_matrix())

    assert {result.case.channel for result in results} >= {
        "plain_text",
        "file_content",
        "hidden_unicode",
        "mcp_metadata",
        "skill_metadata",
        "fake_completion",
    }
    assert all(result.taint_reached for result in results)
    assert all(result.sink_attempted for result in results)
    assert all(not result.sink_fired for result in results)
    assert all(result.blocked for result in results)
    assert all(result.rule_outcome is SinkOutcome.PARTIAL for result in results)


def test_separate_operator_approval_is_visible_as_full_safe_fixture_flow() -> None:
    case = AttackCase(
        case_id="approved-fixture",
        channel="plain_text",
        carrier_type="text",
        payload="CANARY-APPROVED",
        expected_sink="python_execute",
    )

    result = AIGHarness().run_case(case, tainted_sink_approved=True)

    assert result.blocked is False
    assert result.sink_fired is True
    assert result.rule_outcome is SinkOutcome.FULL
    assert result.semantic_outcome is None
