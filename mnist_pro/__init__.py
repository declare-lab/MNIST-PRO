"""MNIST-PRO: an evaluation framework for active-glimpse visual reasoning."""

from .agents import AgentConfig, GlimpseAgent, MEMORY_SPECS
from .env import ActiveGlimpseEnv, TerminationReason
from .matrix import ARMS, HARNESSES, Cell, confounds, discover_runs, load_matrix, status
from .metrics import exploration_stats, stroke_coverage
from .rendering import CanvasSpec, build_canvas, render_composite, render_observation
from .wrappers import TimeLimit, TrajectoryRecorder, make_env

__version__ = "0.1.0"
__all__ = ["ActiveGlimpseEnv", "AgentConfig", "ARMS", "Cell", "CanvasSpec",
           "GlimpseAgent", "HARNESSES", "MEMORY_SPECS", "TerminationReason",
           "TimeLimit", "TrajectoryRecorder", "build_canvas", "confounds",
           "discover_runs", "exploration_stats", "load_matrix", "make_env",
           "render_composite", "render_observation", "status", "stroke_coverage"]
