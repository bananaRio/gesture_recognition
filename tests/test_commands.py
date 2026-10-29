"""Command schema, model-output validation, vetoes and simulator; no models required."""

import pytest

from wave_demo.commands import (ToolCall, Simulator, is_command_discussion, is_compound_request,
                                parse_model_response)


def test_accepts_conversation_and_valid_function():
    assert parse_model_response('{"reply":"Hello!", "action":"none"}') == (None, "Hello!")
    call, reply = parse_model_response('{"reply":"Looking left.","action":"look_left"}')
    assert call.as_dict() == {"name": "look", "arguments": {"direction": "left"}}
    assert reply == "Looking left."
    assert parse_model_response('{"action":"get_status","reply":"Checking."}')[0] == ToolCall("get_status")


@pytest.mark.parametrize("text", [
    'Sure! {"reply":"Hello!","action":"none"}',
    '```json\n{"reply":"Hello!","action":"none"}\n```',
    '{"reply":"Stopping", "action":',
    '{"reply":"Hello!","action":"none","code":"anything"}',
    '{"reply":"","action":"none"}',
    '{"reply":12,"action":"none"}',
    '{"reply":"x","action":"run_python"}',
    '{"reply":"x","action":"look_north"}',
    '{"reply":"x","action":null}',
    '{"reply":"x","action":["stop"]}',
    '{"reply":"x","tool_call":{"name":"exec","arguments":{}}}',
    '[{"reply":"Hello!","action":"none"}]',
    '{"reply":"No action","action":"none","action":"stop"}',
    '{"reply":"No action","reply":"Stop","action":"stop"}',
])
def test_rejects_prose_markdown_truncation_schema_drift_and_duplicate_keys(text):
    with pytest.raises(ValueError):
        parse_model_response(text)


def test_tool_call_validates_names_and_arguments_and_copies_them():
    for name, arguments in (("exec", {}), ("look", {}), ("look", {"direction": "up"}),
                            ("look", {"direction": "left", "speed": "fast"}), ("stop", {"now": "yes"})):
        with pytest.raises(ValueError):
            ToolCall(name, arguments)
    arguments = {"direction": "right"}
    call = ToolCall("look", arguments)
    arguments["direction"] = "left"
    assert call.arguments == {"direction": "right"}


@pytest.mark.parametrize("text", [
    "Do not look left.", "Don’t look right", "Never stop.", "What happens if I say stop?",
    "Explain the command look left.", 'Say "look left" for me.', "Suppose you looked right.",
])
def test_discussions_and_negations_are_vetoed(text):
    assert is_command_discussion(text)


@pytest.mark.parametrize("text", ["Look left.", "Could you turn your head to the right?", "Stop moving, please."])
def test_direct_commands_are_not_vetoed(text):
    assert not is_command_discussion(text)
    assert not is_compound_request(text)


@pytest.mark.parametrize("text", ["Say hello then look right.", "Look left and stop", "Greet them; stop."])
def test_compound_requests_are_vetoed(text):
    assert is_compound_request(text)


def test_simulator_executes_only_validated_calls_and_reports_status():
    simulator = Simulator()
    assert simulator.apply(ToolCall("get_status")) == "I am ready, looking straight ahead."
    assert simulator.apply(ToolCall("look", {"direction": "left"})) == "Looking left."
    assert simulator.apply(ToolCall("greet")) == "Hello! Nice to meet you."
    assert simulator.apply(ToolCall("stop")) == "Stopping."
    assert simulator.apply(ToolCall("get_status")) == "I am stopped, looking left."
    assert simulator.snapshot() == {"direction": "left", "stopped": True, "greetings": 1, "last_action": "stop"}
    assert simulator.apply(ToolCall("look", {"direction": "center"})) == "Looking straight ahead."
    assert simulator.snapshot()["stopped"] is False
    snapshot = simulator.snapshot()
    snapshot["direction"] = "right"
    assert simulator.snapshot()["direction"] == "center"
