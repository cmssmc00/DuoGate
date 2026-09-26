"""Offline checks for source-only guidance and task-driven collection checks."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from appworld_experiments.code.ace import appworld_confidence as control
from appworld_experiments.code.ace.adaptation_agent import StarAgent
from appworld_experiments.code.ace.adaptation_react import SimplifiedReActStarAgent
from appworld_experiments.code.ace.playbook import (
    parse_playbook_line,
    select_playbook_for_task,
)


ROOT = Path(__file__).resolve().parents[2]
INITIAL_PLAYBOOK = ROOT / "experiments/playbooks/appworld_initial_playbook.txt"
GENERATOR_PROMPT = ROOT / "experiments/prompts/appworld_react_generator_prompt.txt"


@pytest.mark.parametrize("task", ["Accept all Splitwise invitations", "Delete all SimpleNote notes"])
def test_initial_playbook_reaches_generator_unchanged(monkeypatch, task):
    # Exercise the real initialization/rendering path without creating model clients.
    monkeypatch.setattr(StarAgent, "initialize", lambda self, world: None)
    agent = SimplifiedReActStarAgent.__new__(SimplifiedReActStarAgent)
    agent.generator_prompt_template = GENERATOR_PROMPT.read_text()
    agent.playbook = INITIAL_PLAYBOOK.read_text()
    agent.enable_playbook_retrieval = True
    agent.max_injected_playbook_chars = 60000
    agent.min_injected_playbook_chars = 15000
    agent.max_injected_playbook_bullets = 80
    agent.max_prompt_length = None
    world = SimpleNamespace(task=SimpleNamespace(
        instruction=task,
        app_descriptions={},
        supervisor={"first_name": "Test", "last_name": "User", "email": "test@example.org"},
        ground_truth=SimpleNamespace(required_apis=[]),
    ))

    agent.initialize(world)

    assert agent.selected_playbook == agent.playbook
    assert len([line for line in agent.selected_playbook.splitlines() if parse_playbook_line(line)]) == 8
    assert agent.playbook_selection_metadata["always_included_count"] == 0
    rendered = "\n".join(message["content"] for message in agent.messages)
    injected = rendered.split("### PLAYBOOK BEGIN\n", 1)[1].split("\n### PLAYBOOK END", 1)[0]
    assert injected == agent.playbook
    assert "Confidence level: LOW / MEDIUM / HIGH" not in rendered
    assert "compact candidate table" not in rendered
    assert "Execution discipline:" not in rendered
    assert rendered.rstrip().endswith(f"Task: {task}")


def test_empty_playbook_does_not_receive_static_guidance():
    selected, metadata = select_playbook_for_task("", "Pay using Venmo")
    assert selected == ""
    assert metadata["selected_playbook_chars"] == 0


def test_oversized_playbook_retrieves_only_existing_entries():
    source = "## API GUIDANCE\n" + "\n".join(
        f"[api-{index:05d}] {'Venmo' if index == 99 else 'Spotify'} evidence {index}: " + "x" * 100
        for index in range(100)
    )
    selected, metadata = select_playbook_for_task(source, "Inspect Venmo requests", max_chars=2000, max_bullets=8)
    source_lines = set(source.splitlines())
    assert all(line in source_lines for line in selected.splitlines() if line)
    assert "[api-00099]" in selected
    assert metadata["retrieval_mode"] == "keyword_selection"
    assert metadata["always_included_count"] == 0
    assert 0 < metadata["selected_bullet_count"] <= 8


def output_messages(rows):
    return [{"role": "user", "content": "Output:\n```\n" + json.dumps(rows) + "\n```"}]


def set_gate(task, code, rows, history=()):
    calls = control.extract_api_calls_from_code(code)
    risk = control.classify_code_risk(calls)["risk"]
    return control._evaluate_set_level_gate(task, code, calls, output_messages(rows) + list(history), risk)


@pytest.mark.parametrize("in_contacts", [False, True])
def test_accept_all_invitations_does_not_depend_on_contact_membership(in_contacts):
    task = "Accept all Splitwise group invitations, including people outside my contact book."
    rows = [{"invitation_code": "invite_001", "in_contacts": in_contacts, "eligible": True}]
    expected = control._extract_recent_expected_set("phone_splitwise_invitations", json.dumps(rows), task)
    excluded = control._extract_explicit_ineligible_set("phone_splitwise_invitations", json.dumps(rows), task)
    assert expected == {"invite_001": "accept"}
    assert excluded == {}
    result = set_gate(task, "apis.splitwise.accept_group_invitation(invitation_code='invite_001')", rows)
    assert result.mode == control.SET_GATE_SHADOW
    assert result.wrong_action_ids == []
    assert result.explicit_ineligible_ids == []


@pytest.mark.parametrize("kind,task,row", [
    ("todoist_tasks", "Review all Todoist tasks", {"task_id": "task_001", "status": "incomplete"}),
    ("venmo_requests", "Review all Venmo requests", {"request_id": "request_001", "status": "pending"}),
    ("phone_splitwise_invitations", "Review all Splitwise invitations", {"invitation_code": "invite_001", "in_contacts": False}),
    ("spotify_playlist_messages", "Review all Spotify songs", {"song_id": "song_001", "suggestion": "remove"}),
    ("simplenote_notes", "Review all SimpleNote notes", {"note_id": "note_001", "title": "monthly habit log"}),
])
def test_app_metadata_does_not_supply_eligibility_or_actions(kind, task, row):
    assert control._extract_recent_expected_set(kind, json.dumps(row), task) == {}
    assert control._extract_explicit_ineligible_set(kind, json.dumps(row), task) == {}
    row["eligible"] = True
    assert set(control._extract_recent_expected_set(kind, json.dumps(row), task).values()) == {"unknown"}


@pytest.mark.parametrize("kind,task,key", [
    ("todoist_tasks", "Delete all Todoist tasks", "task_id"),
    ("simplenote_notes", "Delete all SimpleNote notes", "note_id"),
])
def test_explicit_delete_is_not_replaced_by_app_default_update(kind, task, key):
    row = {key: "item_001", "eligible": True, "planned_action": "update"}
    assert control._extract_recent_expected_set(kind, json.dumps(row), task) == {"item_001": "delete"}


@pytest.mark.parametrize("task", [
    "Handle all Splitwise invitations",
    "Accept all Splitwise invitations if their senders are contacts, otherwise delete them",
    "Delete all Venmo requests and create replacements",
    "Do not update any SimpleNote notes",
])
def test_ambiguous_or_conditional_request_has_no_default_action(task):
    row = {"candidate_id": "item_001", "eligible": True}
    expected = control._extract_recent_expected_set("generic", json.dumps(row), task)
    assert expected == {"item_001": "unknown"}


@pytest.mark.parametrize("task", [
    "Reassign tasks assigned to me in Todoist",
    "Update the playlist accordingly after roommates replied",
    "Review Venmo payment requests from yesterday",
    "Export monthly habit logs from SimpleNote",
])
def test_old_task_phrasings_do_not_force_collection_policy(task):
    assert not control._looks_like_collection_task(task)


def test_adjacent_records_do_not_share_eligibility_or_actions():
    rows = [
        {"note_id": "note_001", "eligible": True, "planned_action": "delete", "skip_reason": ""},
        {"note_id": "note_002", "eligible": False, "skip_reason": "outside requested scope"},
        {"note_id": "note_003", "eligible": True, "planned_action": "update"},
    ]
    text = json.dumps(rows)
    assert control._extract_recent_expected_set("simplenote_notes", text, "Process all SimpleNote notes") == {
        "note_001": "delete", "note_003": "update",
    }
    assert control._extract_explicit_ineligible_set("simplenote_notes", text) == {"note_002": "skip"}


def test_collection_gate_still_blocks_wrong_actions_and_missing_completion():
    task = "Delete all Todoist tasks"
    rows = [{"task_id": item, "eligible": True} for item in ("task_001", "task_002")]
    wrong = set_gate(task, "apis.todoist.update_task(task_id='task_001')", rows)
    assert wrong.mode == control.SET_GATE_HARD_BLOCK_MUTATION
    assert wrong.wrong_action_ids == ["task_001"]

    # A partial correct mutation can proceed, but cannot justify completion.
    code = "apis.todoist.delete_task(task_id='task_001')"
    partial = set_gate(task, code, rows)
    assert partial.mode == control.SET_GATE_SHADOW
    completion = set_gate(task, "apis.supervisor.complete_task()", rows, [
        {"role": "assistant", "content": code},
    ])
    assert completion.mode == control.SET_GATE_HARD_BLOCK_COMPLETE
    assert completion.missing_ids == ["task_002"]


def test_explicit_ineligible_target_is_blocked_without_an_eligible_set():
    result = set_gate("Delete all Todoist tasks", "apis.todoist.delete_task(task_id='task_001')", [
        {"task_id": "task_001", "eligible": False},
    ])
    assert result.mode == control.SET_GATE_HARD_BLOCK_MUTATION
    assert result.explicit_ineligible_ids == ["task_001"]


def test_unknown_action_is_not_a_wrong_action_violation():
    result = set_gate("Process all SimpleNote notes", "apis.simple_note.delete_note(note_id='note_001')", [
        {"note_id": "note_001", "eligible": True},
    ])
    assert result.mode == control.SET_GATE_SHADOW
    assert result.wrong_action_ids == []


def test_dynamic_plan_remains_uncertain_instead_of_being_blocked():
    result = set_gate("Delete all Todoist tasks", "for item in items:\n    apis.todoist.delete_task(task_id=item['id'])", [
        {"task_id": "task_001", "eligible": True},
    ])
    assert result.mode == control.SET_GATE_SHADOW
    assert result.mismatch_type == "uncertain"


def test_assistant_candidate_claims_are_not_treated_as_environment_output():
    code = "apis.todoist.update_task(task_id='task_001')"
    calls = control.extract_api_calls_from_code(code)
    result = control._evaluate_set_level_gate("Process all Todoist tasks", code, calls, [
        {"role": "assistant", "content": '{"task_id": "task_001", "planned_action": "delete"}'},
    ], control.RISK_MUTATION)
    assert result.mode == control.SET_GATE_SHADOW
    assert result.mismatch_type == "uncertain"


def test_runtime_controller_still_intercepts_collection_violations():
    controller = control.AppWorldConfidenceController()
    code = "apis.todoist.update_task(task_id='task_001')"
    result = controller.control(
        proposed_code=code,
        proposed_content=f"Code:\n```python\n{code}\n```",
        messages=output_messages([{"task_id": "task_001", "eligible": True}]),
        task_instruction="Delete all Todoist tasks",
        step_index=1,
    )
    assert result["decision"] == control.DECISION_QUERY_READ_ONLY_API
    assert result["final_code"] != code
    assert result["assessment"]["source"] == "set_level_gate"
    recovery_calls = control.extract_api_calls_from_code(result["final_code"])
    assert control.classify_code_risk(recovery_calls)["risk"] == control.RISK_READ_ONLY
    assert controller._last_set_level_gate.wrong_action_ids == ["task_001"]


def test_paper_payment_direction_check_is_retained():
    assert control._venmo_semantic_api_conflict("send_money", "Request $10 from Alex on Venmo")
    assert control._venmo_semantic_api_conflict("request_money", "Request $10 from Alex on Venmo") is None
