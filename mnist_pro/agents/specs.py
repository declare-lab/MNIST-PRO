"""Memory configurations, declared as data rather than as subclasses.

The previous implementation carried fifteen agent classes whose only differences were
prompt strings and a horizon default -- four memory configurations, crossed with one
and two digits, crossed with natural-conversation variants. Adding a digit count or a
conversation mode meant writing new subclasses in every combination.

Here the taxonomy is four `MemorySpec` values, and digit count and conversation mode
are parameters. The prompt text is carried over verbatim from the original classes so
that runs remain comparable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

BASE_USER_INSTRUCTION = (
    "Based on the observation, output your next move or final answer in the "
    "requested JSON format."
)

THOUGHT_SUFFIX = " Include an extra key 'thought' in the JSON containing your reasoning."

SPATIAL_SUFFIX = (
    " "
    "Assume your starting position is coordinate [0, 0]. Moving up, down, left, or "
    "right changes your relative coordinates by [0, -1], [0, 1], [-1, 0], or [1, 0] "
    "respectively. "
    "Include two extra keys in your JSON response:\n"
    "1. 'thought': your reasoning.\n"
    "2. 'spatial_map': a list of exactly one object representing your current step as "
    "spatial memory, with 'coords': [x, y] and 'features': 'description of what was "
    "observed at those coordinates' (e.g. 'empty background', 'top curve of a digit', "
    "'vertical stroke', etc.).\n"
)


@dataclass(frozen=True)
class MemorySpec:
    """One row of the memory taxonomy."""

    name: str                       # stable key used in configs and paths
    display_name: str               # the name used in the paper
    legacy_class: str               # original class name, for reading old logs
    user_instruction: str
    memory_prompt_prefix: str
    include_text_memory: bool = True     # False = visual buffer only
    default_horizon: int = 1             # images kept; -1 means unbounded
    answer_example_keys: tuple = ()      # extra keys shown in the out-of-steps warning

    def warning_example(self, digits: int) -> str:
        value = '"58"' if digits > 1 else "<digit>"
        parts = []
        if "thought" in self.answer_example_keys:
            parts.append('"thought": "<reasoning>"')
        if "spatial_map" in self.answer_example_keys:
            parts.append('"spatial_map": [{"coords": [x, y], "features": "<observation>"}]')
        parts.append('"action": "answer"')
        parts.append(f'"value": {value}')
        return "{" + ", ".join(parts) + "}"


VISUAL_BUFFER = MemorySpec(
    name="visual_buffer",
    display_name="Multimodal Sensory Memory (Visual Buffer)",
    legacy_class="SensoryVisionAgent",
    user_instruction=BASE_USER_INSTRUCTION,
    memory_prompt_prefix="Previous actions",
    include_text_memory=False,
    default_horizon=4,
)

EVENT_LOGGING = MemorySpec(
    name="event_logging",
    display_name="Implicit Episodic Memory (Event Logging)",
    legacy_class="DefaultVisionAgent",
    user_instruction=BASE_USER_INSTRUCTION,
    memory_prompt_prefix="Previous actions",
    default_horizon=1,
)

TEXTUAL_BELIEF_STATE = MemorySpec(
    name="textual_belief_state",
    display_name="Explicit Working Memory (Textual Belief State)",
    legacy_class="MemoryVisionAgent",
    user_instruction=BASE_USER_INSTRUCTION + THOUGHT_SUFFIX,
    memory_prompt_prefix="Previous actions and thoughts",
    default_horizon=1,
    answer_example_keys=("thought",),
)

METRIC_GRID_MAP = MemorySpec(
    name="metric_grid_map",
    display_name="Structured Spatial Memory (Metric Grid Map)",
    legacy_class="SpatialMemoryVisionAgent",
    user_instruction=BASE_USER_INSTRUCTION + SPATIAL_SUFFIX,
    memory_prompt_prefix="Previous actions, thoughts, and structured spatial map",
    default_horizon=1,
    answer_example_keys=("thought", "spatial_map"),
)

MEMORY_SPECS = {s.name: s for s in
                (VISUAL_BUFFER, EVENT_LOGGING, TEXTUAL_BELIEF_STATE, METRIC_GRID_MAP)}


def system_instruction(digits: int) -> str:
    """Verbatim from BaseVisionAgent / MultiDigitVisionAgent."""
    if digits > 1:
        return (
            "You are an active vision agent playing a game to identify a sequence of "
            "horizontally concatenated MNIST digits. "
            "Your goal is to figure out the exact sequence of digits (e.g. 58) from "
            "left to right. "
            "The digits are drawn in black on a white background, and unseen areas "
            "are masked in dark gray. "
            "You can move the visible box around. "
            "When you are confident about the entire sequence of digits, you must "
            "provide your final answer. "
            "Note: You only have one chance to provide the final answer, so make sure "
            "you are confident before doing so! "
            "For moving, output: {\"action\": \"move\", \"direction\": \"up\"} "
            "(directions: 'up', 'down', 'left', 'right'). "
            "For answering, output the sequence as a raw string of digits, like: "
            '{"action": "answer", "value": "58"}.'
        )
    return (
        "You are an active vision agent playing a game to identify an MNIST digit. "
        "Your goal is to figure out what the digit (0-9) is. "
        "The digit is drawn in black on a white background, and unseen areas are "
        "masked in dark gray. "
        "You can move the visible box around. "
        "When you are confident about the digit, you must provide your final answer. "
        "Note: You only have one chance to provide the final answer, so make sure you "
        "are confident before doing so! "
        "For moving, output: {\"action\": \"move\", \"direction\": \"up\"} "
        "(directions: 'up', 'down', 'left', 'right'). "
        'For answering, output: {"action": "answer", "value": <digit>}.'
    )


def control_prompts(digits: int) -> tuple[str, str]:
    """The unmasked-canvas control. Verbatim from `predict_full_image`.

    Returned by the same module the agents use, so the control can no longer drift
    from the main path -- it used to live in a separate function in each eval driver.
    """
    if digits > 1:
        return (
            "Look at this image of concatenated MNIST digits. What sequence of digits "
            "is shown?",
            "You are a visual recognition model. Your goal is to identify the sequence "
            "of horizontally concatenated MNIST digits shown in the image. "
            "Provide your final decision as a raw string of digits (e.g. '58') in JSON "
            "format at the very end of your response, like: "
            '{"action": "answer", "value": "58"}.',
        )
    return (
        "Look at this image of an MNIST digit. What digit (0-9) is shown?",
        "You are a visual recognition model. Your goal is to identify the MNIST digit "
        "(0-9) shown in the image. "
        "Provide your final decision in JSON format at the very end of your response, "
        'like: {"action": "answer", "value": <digit>}.',
    )


WARNING_TEMPLATE = (
    "You have run out of steps. You must provide your final answer now. Remember, you "
    "only have one chance! Output JSON like {} at the end."
)

# Legacy class name -> (memory spec name, digits, conversation mode). Lets the
# analysis layer read directories produced by the original implementation.
LEGACY_CLASSES = {}
for _spec in MEMORY_SPECS.values():
    LEGACY_CLASSES[_spec.legacy_class] = (_spec.name, 1, "turn_based")
    LEGACY_CLASSES["MultiDigit" + _spec.legacy_class] = (_spec.name, 2, "turn_based")
LEGACY_CLASSES["NaturalConversationVisionAgent"] = ("event_logging", 1, "natural")
LEGACY_CLASSES["MultiDigitNaturalConversationVisionAgent"] = ("event_logging", 2, "natural")
LEGACY_CLASSES["NaturalConversationMemoryVisionAgent"] = ("textual_belief_state", 1, "natural")
LEGACY_CLASSES["MultiDigitNaturalConversationMemoryVisionAgent"] = (
    "textual_belief_state", 2, "natural")
