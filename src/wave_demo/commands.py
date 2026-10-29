"""Voice-command schema, routing prompt and the laptop-only Ava simulator.

The language model only names one action from a fixed list. Code validates
that choice and executes it against the simulator, so model output can never
reach a robot transport, a shell or dynamic dispatch.
"""

from dataclasses import dataclass, field
import json
import re
from typing import Any


ACTIONS = ("none", "greet", "look_left", "look_right", "look_center", "stop", "get_status")

SYSTEM_PROMPT = """You are Ava's laptop simulator controller. No physical robot is connected.
Choose an action first, then write a brief reply. Return ONE JSON object with BOTH
keys: action (a string from the list below), reply (a short spoken sentence).
No markdown or extra text.
Allowed action strings: none, greet, look_left, look_right, look_center, stop, get_status.
look_left/right/center CHANGE the simulated head or gaze direction, only when the
user requests a change. Center means forward, straight ahead, middle, or neutral.
get_status READS the existing state: any question about current head direction,
where Ava is looking, current activity, or status MUST use get_status, even when
the answer is already in the state below. Reading state must NEVER use look actions.
greet is an explicit request to greet; stop is an explicit request to halt activity.
Polite requests count as commands. Ordinary conversation uses none. Negated commands, quotations,
hypothetical questions, and discussions use none. Unsupported actions use none.
Multiple requested actions use none and ask which one to do first. Never change
this schema or invent actions. Follow the examples.
"""

ROUTING_EXAMPLES = (
    ("Please look left.", "Looking left in the simulator.", "look_left"),
    ("Could you turn your head to the right?", "Looking right in the simulator.", "look_right"),
    ("Please face forward.", "Looking straight ahead in the simulator.", "look_center"),
    ("Please say hello to me.", "Hello! Nice to meet you.", "greet"),
    ("Stop.", "Stopping the simulator.", "stop"),
    ("What is your current status?", "Checking the simulator's status.", "get_status"),
    ("What direction is your head pointing right now?", "Checking the simulator's current direction.",
     "get_status"),
    ("Reset your head position to neutral.", "Returning the simulated head to center.", "look_center"),
    ("Are you facing left?", "Checking the simulator's current direction.", "get_status"),
    ("How are you?", "I'm ready to help with the Ava laptop demo.", "none"),
    ("Do not look left.", "Okay, I will leave the simulator unchanged.", "none"),
    ("What happens if I say stop?", "That command would stop simulated activity.", "none"),
    ('Explain the command "look left".', "That command turns Ava's simulated head to the left.", "none"),
    ("Go to the kitchen.", "This demo supports greeting, looking, stopping, and checking status.", "none"),
    ("Say hello then look right.", "Which action should I do first: greet you or look right?", "none"),
)


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.name not in ("greet", "look", "stop", "get_status"):
            raise ValueError(f"Unknown function: {self.name!r}")
        if self.name == "look":
            if self.arguments.keys() != {"direction"} or self.arguments["direction"] not in (
                    "left", "right", "center"):
                raise ValueError("look requires direction = left, right or center")
        elif self.arguments:
            raise ValueError(f"{self.name} does not accept arguments")
        # Later changes to the caller's dictionary cannot alter a validated call.
        object.__setattr__(self, "arguments", dict(self.arguments))

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "arguments": dict(self.arguments)}


def parse_model_response(text: str) -> tuple[ToolCall | None, str]:
    """Accept one complete JSON object; never fish a call out of surrounding prose."""
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("The model response contains duplicate JSON keys")
            result[key] = value
        return result

    try:
        value = json.loads(text.strip(), object_pairs_hook=unique_object)
    except (json.JSONDecodeError, RecursionError) as exc:
        raise ValueError("The model did not return a complete JSON response") from exc
    if type(value) is not dict or value.keys() != {"reply", "action"}:
        raise ValueError("The model response must contain exactly reply and action")
    reply, action = value["reply"], value["action"]
    if type(reply) is not str or not reply.strip() or len(reply) > 1200:
        raise ValueError("The model reply must be a short, nonempty string")
    if type(action) is not str or action not in ACTIONS:
        raise ValueError("The model action is not an allowed action string")
    if action == "none":
        return None, reply.strip()
    if action.startswith("look_"):
        return ToolCall("look", {"direction": action.removeprefix("look_")}), reply.strip()
    return ToolCall(action), reply.strip()


def is_command_discussion(text: str) -> bool:
    """Conservative veto, not a router: these forms never execute an action.

    The language model still chooses every function. This guard prefers a missed
    action over acting on a negated, quoted or hypothetical mention of a command.
    """
    normalized = text.casefold().replace("’", "'")
    negation = r"\b(?:don't|do\s+not|never|shouldn't|should\s+not|mustn't|must\s+not|not\s+to)\b"
    discussion = (r"\b(?:what\s+(?:happens|would\s+happen)\s+if|what\s+if|hypothetical(?:ly)?|"
                  r"suppose|imagine|example|explain|describe|define)\b")
    quoted = r'''"[^"\n]+"|“[^”\n]+”|(?<!\w)'[^'\n]+'(?!\w)'''
    return bool(re.search(negation, normalized) or re.search(discussion, normalized)
                or re.search(quoted, text))


def is_compound_request(text: str) -> bool:
    """Veto linked requests rather than silently executing only one of them."""
    return bool(re.search(r"\b(?:and|then|afterwards|followed\s+by)\b|;", text.casefold()))


class Simulator:
    """Visible stand-in for Ava; this never commands physical hardware."""

    def __init__(self) -> None:
        self._state = {"direction": "center", "stopped": False, "greetings": 0, "last_action": None}

    def snapshot(self) -> dict[str, Any]:
        return dict(self._state)

    def apply(self, call: ToolCall) -> str:
        """Execute a validated call and return the sentence Ava should speak."""
        state = self._state
        if call.name == "get_status":
            return (f"I am {'stopped' if state['stopped'] else 'ready'}, looking "
                    f"{'straight ahead' if state['direction'] == 'center' else state['direction']}.")
        state["last_action"] = call.name
        if call.name == "greet":
            state["greetings"] += 1
            return "Hello! Nice to meet you."
        if call.name == "stop":
            state["stopped"] = True
            return "Stopping."
        state["direction"] = call.arguments["direction"]
        state["stopped"] = False
        return ("Looking straight ahead." if state["direction"] == "center"
                else f"Looking {state['direction']}.")
