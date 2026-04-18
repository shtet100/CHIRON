"""
CHIRON scene awareness.

Maintains a spatial understanding of the workspace: where objects are,
what's stacked on what, which areas are occupied, and how high the arm
needs to lift to avoid collisions during transit.

This is the intelligence layer between perception (BROTEUS / ground truth)
and execution (sequencer). When BROTEUS is connected, it replaces the
data source but this reasoning stays the same.
"""

import numpy as np
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple

from .grasp_computer import ObjectDetection, scan_scene_objects

logger = logging.getLogger("chiron.scene")


@dataclass
class PlacementInfo:
    """Where and how to place an object, accounting for what's already there."""
    position: np.ndarray        # [x, y, z] - actual release point
    stack_height: float         # Total height of stack at target
    objects_below: List[str]    # Names of objects already at this location
    is_stacking: bool           # True if placing on top of another object


@dataclass
class TransitPlan:
    """Safe waypoints for moving between two positions without collision."""
    waypoints: List[np.ndarray]  # List of [x, y, z] points to visit in order
    clearance_height: float      # The safe height used for transit


class SceneAwareness:
    """
    Spatial reasoning about the workspace.

    Before every action, call refresh() to get fresh object positions.
    Then use compute_place_target() and plan_safe_transit() to get
    spatially aware plans.
    """

    def __init__(
        self,
        model=None,
        data=None,
        table_height: float = 0.40,
        clearance_margin: float = 0.08,
        stack_gap: float = 0.005,
        proximity_threshold: float = 0.06,
        gripper_clearance: float = 0.0,
    ):
        """
        Args:
            model: MuJoCo model (for ground truth scanning)
            data: MuJoCo data (for ground truth scanning)
            table_height: Z coordinate of the table surface
            clearance_margin: Extra height above tallest object for safe transit
            stack_gap: Small gap between stacked objects
            proximity_threshold: How close (in XY) two objects need to be
                                 to count as "same location"
            gripper_clearance: How far the lowest gripper point extends below
                               the hand body. The hand body must be this much
                               higher than any object it needs to clear.
                               Measured from gripper geometry, not hardcoded.
        """
        self.model = model
        self.data = data
        self.table_height = table_height
        self.clearance_margin = clearance_margin
        self.stack_gap = stack_gap
        self.proximity_threshold = proximity_threshold
        self.gripper_clearance = gripper_clearance

        # Current scene state
        self.objects: List[ObjectDetection] = []
        self.object_heights: Dict[str, float] = {}

    def refresh(self, detections: List[ObjectDetection] = None):
        """
        Refresh the scene understanding with fresh detections.

        If detections are not provided, reads from MuJoCo ground truth.
        When BROTEUS is connected, pass its detections here instead.
        """
        if detections is not None:
            self.objects = detections
        elif self.model is not None and self.data is not None:
            self.objects = scan_scene_objects(self.model, self.data)
        else:
            logger.warning("No detection source available")
            return

        # Compute object heights from their dimensions
        self.object_heights = {}
        for obj in self.objects:
            self.object_heights[obj.name] = self._get_object_height(obj)

        names = [o.name for o in self.objects]
        positions = {o.name: f"[{o.position[0]:.3f},{o.position[1]:.3f},{o.position[2]:.3f}]"
                     for o in self.objects}
        logger.info(f"Scene refreshed: {len(self.objects)} objects: {positions}")

    def _get_object_height(self, obj: ObjectDetection) -> float:
        """Get the full height of an object from its dimensions."""
        dims = obj.dimensions
        if obj.shape == "box":
            return dims.get("sz", 0.02) * 2  # sz is half-size
        elif obj.shape == "cylinder":
            return dims.get("h", 0.03) * 2  # h is half-height
        elif obj.shape == "sphere":
            return dims.get("r", 0.025) * 2  # diameter
        return 0.04  # Default fallback

    def find_object(self, name: str) -> Optional[ObjectDetection]:
        """Find an object by name in the current scene."""
        for obj in self.objects:
            if obj.name == name:
                return obj
        return None

    def get_objects_near(self, position: np.ndarray, radius: float = None) -> List[ObjectDetection]:
        """Find all objects within radius of a position (XY plane only)."""
        if radius is None:
            radius = self.proximity_threshold
        pos_xy = np.array(position[:2])
        nearby = []
        for obj in self.objects:
            obj_xy = obj.position[:2]
            dist = np.linalg.norm(pos_xy - obj_xy)
            if dist < radius:
                nearby.append(obj)
        return nearby

    def get_objects_above(self, target_name: str) -> List[ObjectDetection]:
        """
        Find objects stacked ABOVE the target in the same stack.

        Returns objects sorted top-to-bottom (the order you'd need to
        remove them to access the target).
        """
        target = self.find_object(target_name)
        if target is None:
            return []

        target_xy = target.position[:2]
        target_z = target.position[2]

        above = []
        for obj in self.objects:
            if obj.name == target_name:
                continue
            # Same XY stack (within proximity threshold)
            dist = np.linalg.norm(obj.position[:2] - target_xy)
            if dist < self.proximity_threshold and obj.position[2] > target_z:
                above.append(obj)

        # Sort top-to-bottom (pick the topmost first)
        above.sort(key=lambda o: o.position[2], reverse=True)
        return above

    def find_clear_spot(self, exclude_positions: List[np.ndarray] = None,
                        min_distance: float = 0.12) -> np.ndarray:
        """
        Find a clear spot on the table to temporarily place objects.

        Searches for a position that's at least min_distance from all
        existing objects and any excluded positions.
        """
        import itertools
        # Candidate positions in a grid within the arm's workspace
        candidates = []
        for x in np.arange(0.30, 0.60, 0.05):
            for y in np.arange(-0.25, 0.25, 0.05):
                candidates.append(np.array([x, y, self.table_height]))

        all_occupied = [obj.position for obj in self.objects]
        if exclude_positions:
            all_occupied.extend(exclude_positions)

        best_pos = None
        best_min_dist = 0

        for cand in candidates:
            min_dist = float('inf')
            for occ in all_occupied:
                d = np.linalg.norm(cand[:2] - occ[:2])
                if d < min_dist:
                    min_dist = d
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_pos = cand

        if best_pos is not None and best_min_dist >= min_distance:
            return best_pos

        # Fallback: just offset from center of table
        return np.array([0.35, 0.20, self.table_height])

    def compute_place_target(
        self,
        target_position: np.ndarray,
        object_being_placed: str,
    ) -> PlacementInfo:
        """
        Compute the actual place position, accounting for stacking.

        If the target zone has objects on it, computes the height to
        stack on top. The position is adjusted to the top of the stack.

        Args:
            target_position: Desired [x, y, z] place location (table level)
            object_being_placed: Name of the object we're about to place

        Returns:
            PlacementInfo with adjusted position and stack context
        """
        target_xy = np.array(target_position[:2])

        # Find what's already at the target location
        objects_at_target = self.get_objects_near(target_position)

        # Filter out the object we're currently holding
        objects_at_target = [o for o in objects_at_target if o.name != object_being_placed]

        if not objects_at_target:
            # Nothing there - place directly on table
            place_z = self.table_height
            return PlacementInfo(
                position=np.array([target_xy[0], target_xy[1], place_z]),
                stack_height=0.0,
                objects_below=[],
                is_stacking=False,
            )

        # Compute stack height: find the highest point among objects at target
        max_top = self.table_height
        objects_below = []
        for obj in objects_at_target:
            obj_top = obj.position[2] + self._get_object_height(obj) / 2
            if obj_top > max_top:
                max_top = obj_top
            objects_below.append(obj.name)

        # Place on top of the stack with a small gap
        place_z = max_top + self.stack_gap
        stack_height = max_top - self.table_height

        # Center XY over the stack (use the topmost object's XY)
        highest_obj = max(objects_at_target, key=lambda o: o.position[2])
        place_x = highest_obj.position[0]
        place_y = highest_obj.position[1]

        logger.info(
            f"Stacking '{object_being_placed}' on top of {objects_below} "
            f"at z={place_z:.3f} (stack height={stack_height:.3f}m)"
        )

        return PlacementInfo(
            position=np.array([place_x, place_y, place_z]),
            stack_height=stack_height,
            objects_below=objects_below,
            is_stacking=True,
        )

    def get_safe_transit_height(self, exclude_object: str = None) -> float:
        """
        Minimum safe height for the HAND BODY so the entire gripper
        (including finger pads below) clears all objects.

        The lowest point of the gripper is hand_z - gripper_clearance.
        For that to clear an object top: hand_z - gripper_clearance > obj_top + margin
        So: hand_z > obj_top + margin + gripper_clearance
        """
        max_top = self.table_height
        for obj in self.objects:
            if obj.name == exclude_object:
                continue
            obj_top = obj.position[2] + self._get_object_height(obj) / 2
            if obj_top > max_top:
                max_top = obj_top

        safe_height = max_top + self.clearance_margin + self.gripper_clearance
        return safe_height

    def _point_to_segment_distance(self, point_xy, seg_start_xy, seg_end_xy):
        """Compute minimum distance from a point to a line segment in 2D."""
        p = np.array(point_xy[:2])
        a = np.array(seg_start_xy[:2])
        b = np.array(seg_end_xy[:2])
        ab = b - a
        ap = p - a
        ab_len_sq = np.dot(ab, ab)
        if ab_len_sq < 1e-10:
            return np.linalg.norm(ap)
        t = np.clip(np.dot(ap, ab) / ab_len_sq, 0.0, 1.0)
        closest = a + t * ab
        return np.linalg.norm(p - closest)

    def get_path_clearance_height(self, start_xy, end_xy, exclude_object: str = None,
                                   corridor_width: float = 0.15) -> float:
        """
        Minimum safe height to transit between two XY positions.

        Only considers objects within the transit corridor (near the path),
        not everything on the table. This means the arm can transit at
        lower heights when the path is clear of obstacles.

        Args:
            start_xy: [x, y] start of transit
            end_xy: [x, y] end of transit
            exclude_object: Object being held (excluded)
            corridor_width: How wide the transit corridor is (arm + object radius)
        """
        max_top = self.table_height
        for obj in self.objects:
            if obj.name == exclude_object:
                continue
            dist = self._point_to_segment_distance(obj.position, start_xy, end_xy)
            if dist < corridor_width:
                obj_top = obj.position[2] + self._get_object_height(obj) / 2
                if obj_top > max_top:
                    max_top = obj_top

        return max_top + self.clearance_margin

    def get_carrying_transit_height(self, start_xy, end_xy, exclude_object: str,
                                     hangover: float, corridor_width: float = 0.15) -> float:
        """
        Minimum safe height for carrying an object along a specific path.

        Uses path-specific clearance: only clears objects near the transit
        corridor, not everything on the table.
        """
        path_safe = self.get_path_clearance_height(
            start_xy, end_xy, exclude_object, corridor_width)
        return path_safe + hangover

    def plan_safe_transit(
        self,
        start_pos: np.ndarray,
        end_pos: np.ndarray,
        held_object: str = None,
    ) -> TransitPlan:
        """
        Plan safe waypoints for moving between two positions.

        Strategy: lift to safe height at start, transit horizontally
        at safe height, then descend at the end. This guarantees the
        arm clears everything on the table.

        Args:
            start_pos: Current EE position [x, y, z]
            end_pos: Target EE position [x, y, z]
            held_object: Object being held (excluded from clearance calc)

        Returns:
            TransitPlan with ordered waypoints
        """
        safe_z = self.get_safe_transit_height(exclude_object=held_object)

        # Ensure we're above safe height at both endpoints
        safe_z = max(safe_z, start_pos[2], end_pos[2])

        waypoints = [
            # 1. Lift to safe height (directly above current position)
            np.array([start_pos[0], start_pos[1], safe_z]),
            # 2. Move horizontally to above the target
            np.array([end_pos[0], end_pos[1], safe_z]),
            # 3. Descend to target (handled by the sequencer's descend phase)
        ]

        return TransitPlan(
            waypoints=waypoints,
            clearance_height=safe_z,
        )

    def get_scene_summary(self) -> dict:
        """Return a human-readable scene summary for the dashboard."""
        summary = {
            "object_count": len(self.objects),
            "objects": [],
            "safe_transit_height": round(self.get_safe_transit_height(), 3),
        }
        for obj in self.objects:
            height = self._get_object_height(obj)
            top_z = obj.position[2] + height / 2
            nearby = self.get_objects_near(obj.position)
            nearby_names = [n.name for n in nearby if n.name != obj.name]
            summary["objects"].append({
                "name": obj.name,
                "shape": obj.shape,
                "position": [round(p, 3) for p in obj.position.tolist()],
                "height": round(height, 3),
                "top_z": round(top_z, 3),
                "nearby": nearby_names,
            })
        return summary
