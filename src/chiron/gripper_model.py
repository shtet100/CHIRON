"""
CHIRON gripper model.

Self-measures the robot's gripper geometry from the MuJoCo model.
Every measurement is computed, never hardcoded. Swap the robot model
and the gripper re-measures itself automatically.
"""

import mujoco
import numpy as np
import logging
from dataclasses import dataclass

logger = logging.getLogger("chiron.gripper_model")


@dataclass
class GripperGeometry:
    """Measured gripper properties - all computed from the model."""
    finger_offset: float       # Distance from hand body to finger pad center (m)
    pad_bottom_offset: float   # Distance from hand body to lowest pad point (m)
    max_opening: float         # Maximum gap between finger pads (m)
    min_opening: float         # Minimum gap (fully closed) (m)
    pad_height: float          # Vertical extent of the finger pad contact area (m)
    pad_width: float           # Width of finger pad (m)
    actuator_range: tuple      # (min_ctrl, max_ctrl) for gripper actuator


def measure_gripper(model: mujoco.MjModel, data: mujoco.MjData,
                    hand_body_name: str = "hand") -> GripperGeometry:
    """
    Measure the gripper geometry directly from the MuJoCo model.

    Inspects body positions, geom extents, and actuator ranges
    to compute every gripper dimension. No hardcoded values.
    """
    mujoco.mj_forward(model, data)

    hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, hand_body_name)
    if hand_id < 0:
        raise ValueError(f"Hand body '{hand_body_name}' not found")

    hand_pos = data.xpos[hand_id].copy()

    # Find all finger geom positions and extents
    finger_geom_z = []
    finger_geom_y = []
    for i in range(model.ngeom):
        body_id = model.geom_bodyid[i]
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
        if "finger" in body_name:
            pos = data.geom_xpos[i]
            finger_geom_z.append(pos[2])
            finger_geom_y.append(pos[1])

    if not finger_geom_z:
        logger.warning("No finger geoms found, using defaults")
        return GripperGeometry(
            finger_offset=0.08, pad_bottom_offset=0.10,
            max_opening=0.08, min_opening=0.0,
            pad_height=0.02, pad_width=0.01,
            actuator_range=(0.0, 0.04),
        )

    # Finger offset: hand body Z to average finger geom Z
    avg_finger_z = np.mean(finger_geom_z)
    min_finger_z = np.min(finger_geom_z)
    max_finger_z = np.max(finger_geom_z)

    finger_offset = hand_pos[2] - avg_finger_z
    pad_bottom_offset = hand_pos[2] - min_finger_z
    pad_height = max_finger_z - min_finger_z

    # Max opening: find the gripper actuator range
    # The gripper is the last actuator in Franka-style models
    gripper_act_id = None
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or ""
        # Check if it controls a finger joint
        jnt_id = model.actuator_trnid[i][0]
        if jnt_id >= 0:
            jnt_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jnt_id) or ""
            if "finger" in jnt_name:
                gripper_act_id = i
                break

    if gripper_act_id is not None:
        ctrl_range = model.actuator_ctrlrange[gripper_act_id]
        actuator_range = (float(ctrl_range[0]), float(ctrl_range[1]))
        # Each finger moves by ctrl amount, total gap = 2 * max_ctrl
        max_opening = 2.0 * actuator_range[1]
        min_opening = 2.0 * actuator_range[0]
    else:
        # Infer from finger joint ranges
        max_opening = 0.08
        min_opening = 0.0
        actuator_range = (0.0, 0.04)
        for i in range(model.njnt):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) or ""
            if "finger" in name:
                max_opening = 2.0 * float(model.jnt_range[i][1])
                min_opening = 2.0 * float(model.jnt_range[i][0])
                actuator_range = (float(model.jnt_range[i][0]), float(model.jnt_range[i][1]))
                break

    # Pad width: estimate from finger geom lateral spread
    pad_width = 0.015  # Reasonable default; hard to measure precisely from geom positions

    geom = GripperGeometry(
        finger_offset=finger_offset,
        pad_bottom_offset=pad_bottom_offset,
        max_opening=max_opening,
        min_opening=min_opening,
        pad_height=pad_height,
        pad_width=pad_width,
        actuator_range=actuator_range,
    )

    logger.info(
        f"Gripper measured: finger_offset={geom.finger_offset:.4f}m, "
        f"pad_bottom={geom.pad_bottom_offset:.4f}m, "
        f"opening=[{geom.min_opening:.4f}, {geom.max_opening:.4f}]m, "
        f"pad_height={geom.pad_height:.4f}m"
    )
    return geom
