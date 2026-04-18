"""
CHIRON backend abstraction layer.

Every robot (simulated or physical) implements this interface.
CHIRON's server code never touches simulator or hardware APIs directly.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional
import time


@dataclass
class RobotInfo:
    """Static robot description. Doesn't change after connection."""
    name: str
    dof: int
    joint_names: List[str]
    joint_limits_lower: List[float]
    joint_limits_upper: List[float]
    has_gripper: bool = False
    gripper_range: tuple = (0.0, 1.0)


@dataclass
class RobotState:
    """Snapshot of robot state at a single instant."""
    timestamp: float
    joint_positions: List[float]
    joint_velocities: List[float]
    joint_efforts: List[float]
    gripper_state: float = 0.0  # 0.0 = closed, 1.0 = open
    end_effector_position: Optional[List[float]] = None  # [x, y, z]
    end_effector_orientation: Optional[List[float]] = None  # [x, y, z, w] quaternion
    trajectory_active: bool = False
    trajectory_progress: float = 0.0  # 0.0 to 1.0

    def to_dict(self) -> dict:
        d = {
            "timestamp": self.timestamp,
            "joint_positions": self.joint_positions,
            "joint_velocities": self.joint_velocities,
            "joint_efforts": self.joint_efforts,
            "gripper_state": self.gripper_state,
            "trajectory_active": self.trajectory_active,
            "trajectory_progress": self.trajectory_progress,
        }
        if self.end_effector_position is not None:
            d["end_effector_pose"] = {
                "position": self.end_effector_position,
                "orientation": self.end_effector_orientation or [0, 0, 0, 1],
            }
        return d


class RobotBackend(ABC):
    """
    Abstract base class for all robot backends.

    Implementations:
        MuJoCoBackend  - CPU simulation via mujoco Python bindings
        GenesisBackend - GPU simulation via Genesis (Vulkan)
        LeRobotBackend - Physical SO-ARM 101 via Dynamixel
        ROS2ControlBackend - Any ros2_control-compatible robot
    """

    @abstractmethod
    def connect(self) -> bool:
        """Initialize connection to robot/sim. Returns True on success."""
        ...

    @abstractmethod
    def disconnect(self) -> None:
        """Gracefully shut down connection."""
        ...

    @abstractmethod
    def get_state(self) -> RobotState:
        """Read current robot state. Must be thread-safe."""
        ...

    @abstractmethod
    def send_joint_positions(self, positions: List[float], duration: float) -> bool:
        """Command arm joints to target positions over duration seconds."""
        ...

    @abstractmethod
    def set_gripper(self, openness: float) -> bool:
        """Set gripper openness: 0.0 = closed, 1.0 = open."""
        ...

    @abstractmethod
    def emergency_stop(self) -> bool:
        """Immediate halt. Freeze all joints at current positions."""
        ...

    @abstractmethod
    def get_info(self) -> RobotInfo:
        """Return static robot description."""
        ...

    @abstractmethod
    def step(self) -> None:
        """Advance simulation by one timestep. No-op for physical backends."""
        ...
