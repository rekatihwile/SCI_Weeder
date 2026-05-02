from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class RunContext:
    gantry: Any = None
    cameras: Any = None
    detector: Any = None
    coarse_mover: Any = None
    logger: Any = None
    left_detections: List[Any] = field(default_factory=list)
    right_detections: List[Any] = field(default_factory=list)
    matched_targets: List[Dict[str, Any]] = field(default_factory=list)
    solved_targets: List[Dict[str, Any]] = field(default_factory=list)
    planned_targets: List[Dict[str, Any]] = field(default_factory=list)
    actual_hits: List[Dict[str, Any]] = field(default_factory=list)
    recording_dir: Optional[Any] = None
    warmup_info: Dict[str, Any] = field(default_factory=dict)
    model_load_time_s: float = 0.0

