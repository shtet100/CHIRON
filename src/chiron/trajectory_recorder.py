"""
CHIRON trajectory recorder.

Passively records joint states during execution for DAEDALUS.
Records: timestamp, joint positions, velocities, efforts, EE pose,
gripper state, and the active sequencer phase.

Data is stored in memory and can be exported as JSON or numpy arrays.
"""

import time
import logging
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict

logger = logging.getLogger("chiron.recorder")


@dataclass
class TrajectoryFrame:
    """Single frame of trajectory data."""
    timestamp: float
    joint_positions: List[float]
    joint_velocities: List[float]
    joint_efforts: List[float]
    ee_position: Optional[List[float]]
    ee_orientation: Optional[List[float]]
    gripper_state: float
    phase: str


class TrajectoryRecorder:
    """
    Records trajectory data during pick-and-place execution.

    Start recording before a sequence, stop after. The recorded
    data is what DAEDALUS will use for physics discovery.
    """

    def __init__(self, backend, max_frames: int = 50000):
        self.backend = backend
        self.max_frames = max_frames
        self.frames: List[TrajectoryFrame] = []
        self._recording = False
        self._current_phase = "idle"

    @property
    def is_recording(self) -> bool:
        return self._recording

    def start(self):
        self.frames = []
        self._recording = True
        logger.info("Trajectory recording started")

    def stop(self):
        self._recording = False
        logger.info(f"Trajectory recording stopped: {len(self.frames)} frames")

    def set_phase(self, phase: str):
        self._current_phase = phase

    def record_frame(self):
        """Record one frame from the current backend state."""
        if not self._recording:
            return
        if len(self.frames) >= self.max_frames:
            return

        state = self.backend.get_state()
        frame = TrajectoryFrame(
            timestamp=state.timestamp,
            joint_positions=state.joint_positions[:],
            joint_velocities=state.joint_velocities[:],
            joint_efforts=state.joint_efforts[:],
            ee_position=state.end_effector_position[:] if state.end_effector_position else None,
            ee_orientation=state.end_effector_orientation[:] if state.end_effector_orientation else None,
            gripper_state=state.gripper_state,
            phase=self._current_phase,
        )
        self.frames.append(frame)

    def to_dict(self) -> dict:
        """Export as a dictionary (JSON-serializable)."""
        return {
            "frame_count": len(self.frames),
            "duration": self.frames[-1].timestamp - self.frames[0].timestamp if len(self.frames) > 1 else 0,
            "frames": [
                {
                    "t": f.timestamp,
                    "q": f.joint_positions,
                    "dq": f.joint_velocities,
                    "tau": f.joint_efforts,
                    "ee_pos": f.ee_position,
                    "ee_quat": f.ee_orientation,
                    "grip": f.gripper_state,
                    "phase": f.phase,
                }
                for f in self.frames
            ],
        }

    def summary(self) -> dict:
        """Brief summary without all frames."""
        if not self.frames:
            return {"frame_count": 0, "recording": self._recording}
        return {
            "frame_count": len(self.frames),
            "duration": round(self.frames[-1].timestamp - self.frames[0].timestamp, 2),
            "recording": self._recording,
            "phases_seen": list(set(f.phase for f in self.frames)),
        }
