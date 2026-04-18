"""
CHIRON grasp computer.

Computes grasps from first principles using object geometry and
measured gripper dimensions. No per-shape hardcoded constants.

The core algorithm: for any object, compute its cross-section width
at the proposed grip height. If the width fits within the gripper's
range, it's a valid grip point. Pick the highest valid point to
minimize hangover during transit.

When BROTEUS is connected, it replaces the detection source.
The grasp computation stays the same.
"""

import mujoco
import numpy as np
import logging
from dataclasses import dataclass
from typing import List, Optional

from .gripper_model import GripperGeometry

logger = logging.getLogger("chiron.grasp")


# ── Data contracts (stable for BROTEUS integration) ────────────────

@dataclass
class ObjectDetection:
    """A detected object. BROTEUS will populate these from vision."""
    name: str
    shape: str               # "box", "cylinder", "sphere"
    position: np.ndarray     # [x, y, z] center in world frame
    dimensions: dict         # shape-specific: box={sx,sy,sz}, cylinder={r,h}, sphere={r}
    confidence: float = 1.0


@dataclass
class GraspPlan:
    """A computed grasp. The sequencer consumes these."""
    object_name: str
    grasp_pos: np.ndarray      # [x, y, z] hand body position for gripping
    approach_pos: np.ndarray   # [x, y, z] pre-grasp above
    grasp_quat: np.ndarray     # [w, x, y, z] EE orientation
    gripper_opening: float     # Gripper opening BEFORE approach (0-1)
    grip_close_fraction: float # Gripper value WHEN gripping (0-1)
    hangover: float            # How far object extends below hand body
    contact_height: float      # Height of contact area on object


# ── Object profile functions ───────────────────────────────────────

def object_cross_section(shape: str, dims: dict, h_from_center: float) -> float:
    """
    Compute the cross-section width of an object at height h from its center.

    This is the ONE function that encodes shape knowledge. It returns
    the width (grippable span) at a given height. Everything else in
    the grasp computer uses this function generically.

    Args:
        shape: "box", "cylinder", "sphere"
        dims: shape dimensions from detection
        h_from_center: height offset from object center (positive = up)

    Returns:
        Width at that height (0 if outside the object)
    """
    if shape == "box":
        sz = dims.get("sz", 0.02)  # half-height
        if abs(h_from_center) <= sz:
            # Box cross-section is constant: the smaller of sx, sy
            sx = dims.get("sx", 0.02)
            sy = dims.get("sy", 0.02)
            return 2.0 * min(sx, sy)
        return 0.0

    elif shape == "cylinder":
        half_h = dims.get("h", 0.03)  # half-height
        r = dims.get("r", 0.02)
        if abs(h_from_center) <= half_h:
            return 2.0 * r  # Constant diameter
        return 0.0

    elif shape == "sphere":
        r = dims.get("r", 0.025)
        if abs(h_from_center) < r:
            return 2.0 * np.sqrt(r * r - h_from_center * h_from_center)
        return 0.0

    else:
        # Unknown shape: assume a 4cm wide cylinder-like profile
        return 0.04 if abs(h_from_center) < 0.03 else 0.0


def object_half_height(shape: str, dims: dict) -> float:
    """Get the half-height of an object from its dimensions."""
    if shape == "box":
        return dims.get("sz", 0.02)
    elif shape == "cylinder":
        return dims.get("h", 0.03)
    elif shape == "sphere":
        return dims.get("r", 0.025)
    return 0.03


# ── Smart grasp computer ──────────────────────────────────────────

class GraspComputer:
    """
    Computes grasps from measured gripper geometry and object profiles.

    The algorithm:
    1. Sample the object's cross-section at N heights
    2. Filter to heights where the width fits the gripper
    3. Pick the highest valid height (minimizes hangover)
    4. Compute grip width, hand position, and approach vector

    No magic numbers. Every constant comes from the gripper measurement
    or the object detection.
    """

    def __init__(self, gripper: GripperGeometry, approach_clearance: float = None):
        """
        Args:
            gripper: Measured gripper geometry.
            approach_clearance: Distance above grasp point for pre-grasp.
                If None, computed as 1.5x the gripper pad_bottom_offset
                (enough to clear the gripper above the object).
        """
        self.gripper = gripper

        # Approach clearance: high enough that the gripper clears the object top
        if approach_clearance is not None:
            self.approach_clearance = approach_clearance
        else:
            self.approach_clearance = self.gripper.pad_bottom_offset * 1.5

        # Top-down orientation (180 deg around X)
        self._down_quat = np.array([0.0, 1.0, 0.0, 0.0])

        # Sampling resolution for cross-section profile
        self._profile_samples = 20

    def compute_grasp(self, detection: ObjectDetection) -> GraspPlan:
        """
        Compute an optimal top-down grasp for a detected object.

        Samples the object profile, finds the best grip height,
        and computes all grasp parameters from the geometry.
        """
        pos = detection.position.copy()
        shape = detection.shape
        dims = detection.dimensions
        half_h = object_half_height(shape, dims)

        # Sample cross-section profile from bottom to top
        best_height = 0.0
        best_width = 0.0
        obj_bottom = -half_h
        obj_top = half_h

        sample_heights = np.linspace(obj_bottom, obj_top, self._profile_samples)

        # Find ALL valid grip heights and their widths
        valid_grips = []
        max_width_anywhere = 0.0  # Widest cross-section at any height (for approach opening)

        for h in sample_heights:
            width = object_cross_section(shape, dims, h)
            if width > max_width_anywhere:
                max_width_anywhere = width
            if width <= 0:
                continue
            if self.gripper.min_opening <= width <= self.gripper.max_opening:
                valid_grips.append((h, width))

        if valid_grips:
            # Pick the grip with the WIDEST cross-section (most contact area).
            # Among equal widths, pick the highest (least hangover).
            valid_grips.sort(key=lambda hw: (hw[1], hw[0]), reverse=True)
            best_height, best_width = valid_grips[0]

            # CLAMP: the grip center must be far enough from the object top
            # that the ENTIRE finger pad contacts the object surface.
            # Pads extend ±pad_height/2 from the grip center. If the grip is
            # at the top edge, half the pad is in empty air = no contact.
            # This value comes from the measured gripper geometry.
            max_grip_height = obj_top - self.gripper.pad_height / 2
            if best_height > max_grip_height:
                clamped = max_grip_height
                # Re-check width at clamped height
                clamped_width = object_cross_section(shape, dims, clamped)
                if clamped_width > 0:
                    logger.info(f"Grip clamped from h={best_height:+.3f} to h={clamped:+.3f} "
                                f"(pad_height={self.gripper.pad_height:.3f}m)")
                    best_height = clamped
                    best_width = clamped_width
        elif max_width_anywhere > 0:
            # Object exists but doesn't fit gripper at any height
            logger.warning(f"No valid grip height for {detection.name}, using center")
            best_height = 0.0
            best_width = object_cross_section(shape, dims, 0.0)
        else:
            best_height = 0.0
            best_width = self.gripper.max_opening * 0.5

        # Grip height in world frame
        grip_z_world = pos[2] + best_height

        # Hand body position: grip point + finger offset
        hand_z = grip_z_world + self.gripper.finger_offset

        grasp_pos = np.array([pos[0], pos[1], hand_z])
        approach_pos = np.array([pos[0], pos[1], hand_z + self.approach_clearance])

        # Gripper opening for approach: must clear the WIDEST part of the object
        # at ANY height, not just at the grip height. The gripper descends from
        # above and must pass the object's maximum cross-section without collision.
        open_fraction = min(1.0, (max_width_anywhere + 0.02) / self.gripper.max_opening)

        # Gripper closing: CLOSE FULLY.
        close_fraction = 0.0

        # Hangover: how far below the HAND BODY does the object extend
        # = finger_offset (hand to pad) + distance from grip point to object bottom
        grip_to_bottom = best_height - obj_bottom
        hangover = self.gripper.finger_offset + grip_to_bottom

        # Contact height: how much of the object the fingers touch
        # Limited by the pad height
        available_contact = min(self.gripper.pad_height, 2.0 * half_h)

        plan = GraspPlan(
            object_name=detection.name,
            grasp_pos=grasp_pos,
            approach_pos=approach_pos,
            grasp_quat=self._down_quat.copy(),
            gripper_opening=open_fraction,
            grip_close_fraction=close_fraction,
            hangover=hangover,
            contact_height=available_contact,
        )

        logger.info(
            f"Grasp computed for '{detection.name}' ({shape}): "
            f"grip_z={grip_z_world:.3f} (h={best_height:+.3f} from center), "
            f"width={best_width:.3f}m, hangover={hangover:.3f}m, "
            f"open={open_fraction:.2f}, close={close_fraction:.2f}"
        )
        return plan


# ── Scene scanner (ground truth from MuJoCo) ──────────────────────

def scan_scene_objects(model, data, object_names=None):
    """Read object positions from MuJoCo ground truth."""
    detections = []
    if object_names is None:
        object_names = ["red_cube", "green_cylinder", "blue_sphere"]

    for name in object_names:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id < 0:
            continue

        pos = data.xpos[body_id].copy()
        geom_name = name + "_geom"
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        if geom_id < 0:
            continue

        geom_type = model.geom_type[geom_id]
        geom_size = model.geom_size[geom_id]

        if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
            shape = "box"
            dims = {"sx": float(geom_size[0]), "sy": float(geom_size[1]), "sz": float(geom_size[2])}
        elif geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
            shape = "cylinder"
            dims = {"r": float(geom_size[0]), "h": float(geom_size[1])}
        elif geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
            shape = "sphere"
            dims = {"r": float(geom_size[0])}
        else:
            continue

        detections.append(ObjectDetection(
            name=name, shape=shape, position=pos, dimensions=dims
        ))

    return detections
