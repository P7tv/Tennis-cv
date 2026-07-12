"""Loeuf CV Pipeline — tennis biomechanical measurement engine.

Phase 1: single-stroke inference (stroke type known, back-view video)
Phase 2: full-session inference via motion-energy action spotting
"""

__version__ = "0.2.0"

from .config import PipelineConfig
from .stroke_pipeline import StrokePipeline
from .session_pipeline import SessionPipeline

__all__ = ["PipelineConfig", "StrokePipeline", "SessionPipeline", "__version__"]
