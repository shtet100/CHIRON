"""
MuJoCo simulation backend for CHIRON.

Loads an MJCF model, steps the physics at a configurable rate,
and exposes the standard RobotBackend interface. Thread-safe:
the sim loop runs in its own thread, and get_state() / send_*()
are guarded by a Lock.
"""

import mujoco
import numpy as np
import time
import threading
import logging
from typing import List, Optional

from .base import RobotBackend, RobotState, RobotInfo

logger = logging.getLogger("chiron.mujoco")


class MuJoCoBackend(RobotBackend):
    def __init__(self, model_path: str, dt: float = 0.002):
        self.model_path = model_path
        self.dt = dt

        # MuJoCo objects (set on connect)
        self.model: Optional[mujoco.MjModel] = None
        self.data: Optional[mujoco.MjData] = None

        # Thread safety
        self._lock = threading.Lock()
        self._connected = False

        # Arm vs gripper split (set on connect based on model inspection)
        self._arm_actuator_ids: List[int] = []
        self._gripper_actuator_id: Optional[int] = None
        self._arm_joint_ids: List[int] = []
        self._ee_body_id: Optional[int] = None

        # Target state
        self._target_positions: Optional[np.ndarray] = None
        self._gripper_target: float = 0.0

        # Trajectory interpolation
        self._traj_start_positions: Optional[np.ndarray] = None
        self._traj_end_positions: Optional[np.ndarray] = None
        self._traj_start_time: float = 0.0
        self._traj_duration: float = 0.0
        self._traj_active: bool = False

    def connect(self) -> bool:
        try:
            self.model = mujoco.MjModel.from_xml_path(self.model_path)
            self.data = mujoco.MjData(self.model)

            # Override timestep to match our config
            self.model.opt.timestep = self.dt

            # Identify arm joints vs finger joints vs object freejoints
            # Only hinge joints (type 3) that aren't fingers/grippers
            arm_joints = []
            for i in range(self.model.njnt):
                name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i) or ""
                jnt_type = self.model.jnt_type[i]
                if jnt_type != mujoco.mjtJoint.mjJNT_HINGE:
                    continue  # Skip freejoints, ball joints, slide joints
                if name.startswith("finger") or name.startswith("gripper"):
                    continue  # Skip gripper joints
                arm_joints.append(i)
            self._arm_joint_ids = arm_joints

            # Identify actuators: last actuator is gripper if model has fingers
            has_fingers = any(
                (mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i) or "").startswith("finger")
                for i in range(self.model.njnt)
            )
            if has_fingers and self.model.nu > 1:
                self._arm_actuator_ids = list(range(self.model.nu - 1))
                self._gripper_actuator_id = self.model.nu - 1
            else:
                self._arm_actuator_ids = list(range(self.model.nu))
                self._gripper_actuator_id = None

            # Find end-effector body for FK
            # Try common EE body names
            for ee_name in ["hand", "end_effector", "tool0", "ee_link", "link8"]:
                ee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, ee_name)
                if ee_id >= 0:
                    self._ee_body_id = ee_id
                    break

            # Initialize targets to current positions
            mujoco.mj_forward(self.model, self.data)
            n_arm = len(self._arm_actuator_ids)
            self._target_positions = np.array(
                [self.data.qpos[self._arm_joint_ids[i]] for i in range(n_arm)]
            )
            self.data.ctrl[self._arm_actuator_ids] = self._target_positions

            self._connected = True
            logger.info(
                f"Connected to MuJoCo model: {self.model_path} "
                f"({len(self._arm_actuator_ids)} arm actuators, "
                f"gripper={'yes' if self._gripper_actuator_id is not None else 'no'}, "
                f"EE body={'found' if self._ee_body_id else 'not found'})"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to connect to MuJoCo model: {e}")
            return False

    def disconnect(self) -> None:
        with self._lock:
            self._connected = False
            self.model = None
            self.data = None
            logger.info("MuJoCo backend disconnected")

    def get_state(self) -> RobotState:
        with self._lock:
            if not self._connected:
                return RobotState(
                    timestamp=time.time(),
                    joint_positions=[], joint_velocities=[], joint_efforts=[]
                )

            n_arm = len(self._arm_actuator_ids)
            positions = [float(self.data.qpos[self._arm_joint_ids[i]]) for i in range(n_arm)]
            velocities = [float(self.data.qvel[self._arm_joint_ids[i]]) for i in range(n_arm)]

            # Actuator forces for arm joints
            efforts = [float(self.data.actuator_force[i]) for i in self._arm_actuator_ids]

            # Gripper state: read finger joint position, normalize to 0-1
            gripper = 0.0
            if self._gripper_actuator_id is not None:
                # Franka fingers: range [0, 0.04], normalize
                finger_joints = [
                    i for i in range(self.model.njnt)
                    if (mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i) or "").startswith("finger")
                ]
                if finger_joints:
                    finger_pos = float(self.data.qpos[finger_joints[0]])
                    finger_range = self.model.jnt_range[finger_joints[0]]
                    if finger_range[1] > finger_range[0]:
                        gripper = float((finger_pos - finger_range[0]) / (finger_range[1] - finger_range[0]))

            # End-effector pose via FK
            ee_pos = None
            ee_quat = None
            if self._ee_body_id is not None:
                ee_pos = self.data.xpos[self._ee_body_id].tolist()
                # MuJoCo quaternion is [w, x, y, z], ROS 2 convention is [x, y, z, w]
                mj_quat = self.data.xquat[self._ee_body_id]
                ee_quat = [float(mj_quat[1]), float(mj_quat[2]), float(mj_quat[3]), float(mj_quat[0])]

            # Trajectory progress
            traj_active = self._traj_active
            traj_progress = 0.0
            if traj_active:
                elapsed = time.time() - self._traj_start_time
                traj_progress = min(1.0, elapsed / self._traj_duration)

            return RobotState(
                timestamp=time.time(),
                joint_positions=positions,
                joint_velocities=velocities,
                joint_efforts=efforts,
                gripper_state=gripper,
                end_effector_position=ee_pos,
                end_effector_orientation=ee_quat,
                trajectory_active=traj_active,
                trajectory_progress=traj_progress,
            )

    def send_joint_positions(self, positions: List[float], duration: float) -> bool:
        with self._lock:
            if not self._connected:
                return False
            n_arm = len(self._arm_actuator_ids)
            if len(positions) != n_arm:
                logger.warning(f"Expected {n_arm} joint positions, got {len(positions)}")
                return False

            # Validate joint limits
            for i, pos in enumerate(positions):
                jnt_id = self._arm_joint_ids[i]
                lo, hi = self.model.jnt_range[jnt_id]
                if pos < lo or pos > hi:
                    logger.warning(
                        f"Joint {i} position {pos:.3f} outside limits [{lo:.3f}, {hi:.3f}]"
                    )
                    return False

            # Capture current positions as trajectory start
            self._traj_start_positions = np.array(
                [self.data.qpos[self._arm_joint_ids[i]] for i in range(n_arm)]
            )
            self._traj_end_positions = np.array(positions[:n_arm])
            self._traj_start_time = time.time()
            self._traj_duration = max(duration, 0.01)  # minimum 10ms
            self._traj_active = True
            self._target_positions = self._traj_end_positions

            return True

    def set_gripper(self, openness: float) -> bool:
        with self._lock:
            if not self._connected or self._gripper_actuator_id is None:
                return False
            # Clamp to [0, 1] then scale to actuator range
            openness = max(0.0, min(1.0, openness))
            actuator_id = self._gripper_actuator_id
            ctrl_range = self.model.actuator_ctrlrange[actuator_id]
            ctrl_val = ctrl_range[0] + openness * (ctrl_range[1] - ctrl_range[0])
            self.data.ctrl[actuator_id] = ctrl_val
            self._gripper_target = openness
            return True

    def emergency_stop(self) -> bool:
        with self._lock:
            if not self._connected:
                return False
            # Cancel any active trajectory
            self._traj_active = False
            # Freeze at current positions
            n_arm = len(self._arm_actuator_ids)
            current = np.array([self.data.qpos[self._arm_joint_ids[i]] for i in range(n_arm)])
            self._target_positions = current
            self.data.ctrl[self._arm_actuator_ids] = current
            # Zero velocities
            for jid in self._arm_joint_ids:
                self.data.qvel[jid] = 0.0
            logger.warning("EMERGENCY STOP executed")
            return True

    def get_info(self) -> RobotInfo:
        if not self._connected:
            return RobotInfo(name="disconnected", dof=0, joint_names=[],
                             joint_limits_lower=[], joint_limits_upper=[])

        n_arm = len(self._arm_actuator_ids)
        joint_names = []
        limits_lo = []
        limits_hi = []
        for i in range(n_arm):
            jnt_id = self._arm_joint_ids[i]
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, jnt_id) or f"joint_{i}"
            joint_names.append(name)
            limits_lo.append(float(self.model.jnt_range[jnt_id, 0]))
            limits_hi.append(float(self.model.jnt_range[jnt_id, 1]))

        return RobotInfo(
            name=self.model.names.split(b"\x00")[0].decode() or "mujoco_robot",
            dof=n_arm,
            joint_names=joint_names,
            joint_limits_lower=limits_lo,
            joint_limits_upper=limits_hi,
            has_gripper=self._gripper_actuator_id is not None,
        )

    @staticmethod
    def _quintic_smoothstep(t: float) -> float:
        """
        Quintic smoothstep: s(t) = 6t^5 - 15t^4 + 10t^3

        Maps t in [0,1] to s in [0,1] with zero velocity and zero
        acceleration at both endpoints. This gives the arm a natural
        ease-in / ease-out motion profile, the same shape a real
        servo trajectory planner would use.
        """
        t = max(0.0, min(1.0, t))
        return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)

    def step(self) -> None:
        with self._lock:
            if not self._connected:
                return

            # If a trajectory is active, interpolate and write to ctrl
            if self._traj_active:
                elapsed = time.time() - self._traj_start_time
                t = elapsed / self._traj_duration

                if t >= 1.0:
                    # Trajectory complete: snap to final target
                    self.data.ctrl[self._arm_actuator_ids] = self._traj_end_positions
                    self._traj_active = False
                else:
                    # Interpolate using quintic smoothstep
                    s = self._quintic_smoothstep(t)
                    interpolated = (
                        self._traj_start_positions
                        + s * (self._traj_end_positions - self._traj_start_positions)
                    )
                    self.data.ctrl[self._arm_actuator_ids] = interpolated

            mujoco.mj_step(self.model, self.data)
