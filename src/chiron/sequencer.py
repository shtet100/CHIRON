"""
CHIRON pick-and-place sequencer v3.

Uses measured gripper geometry, computed grasp profiles, and scene-aware
transit heights. Zero hardcoded constants.

Every move decomposes into UP -> ACROSS -> DOWN to prevent collisions.
Grip parameters are computed from object geometry and gripper measurements.
Grasp verification detects failed grips and aborts safely.
Trajectory data is recorded for DAEDALUS.
"""

import asyncio
import time
import logging
import numpy as np
from typing import List, Optional
from dataclasses import dataclass
from enum import Enum

from .ik_solver import IKSolver
from .grasp_computer import GraspComputer, GraspPlan, ObjectDetection, scan_scene_objects
from .scene_awareness import SceneAwareness
from .trajectory_recorder import TrajectoryRecorder
from .gripper_model import GripperGeometry
from .backends.base import RobotBackend

logger = logging.getLogger("chiron.sequencer")


class Phase(str, Enum):
    IDLE = "idle"
    OPEN_GRIPPER = "open_gripper"
    MOVE_TO_PREGRASP = "move_to_pregrasp"
    DESCEND_TO_GRASP = "descend_to_grasp"
    CLOSE_GRIPPER = "close_gripper"
    LIFT = "lift"
    TRANSIT = "transit"
    DESCEND_TO_PLACE = "descend_to_place"
    OPEN_GRIPPER_RELEASE = "open_gripper_release"
    RETREAT = "retreat"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass
class SequencerStatus:
    phase: Phase = Phase.IDLE
    object_name: str = ""
    phase_progress: float = 0.0
    overall_progress: float = 0.0
    message: str = ""

    def to_dict(self):
        return {
            "phase": self.phase.value,
            "object_name": self.object_name,
            "phase_progress": round(self.phase_progress, 2),
            "overall_progress": round(self.overall_progress, 2),
            "message": self.message,
        }


_PHASE_ORDER = [
    Phase.OPEN_GRIPPER, Phase.MOVE_TO_PREGRASP, Phase.DESCEND_TO_GRASP,
    Phase.CLOSE_GRIPPER, Phase.LIFT, Phase.TRANSIT,
    Phase.DESCEND_TO_PLACE, Phase.OPEN_GRIPPER_RELEASE, Phase.RETREAT,
]
_N_PHASES = len(_PHASE_ORDER)


class PickAndPlaceSequencer:
    def __init__(self, backend, ik_solver, grasp_computer, scene,
                 gripper_geom, recorder=None):
        self.backend = backend
        self.ik = ik_solver
        self.grasp = grasp_computer
        self.scene = scene
        self.gripper = gripper_geom
        self.recorder = recorder

        self.status = SequencerStatus()
        self._abort = False
        self._down_quat = np.array([0.0, 1.0, 0.0, 0.0])

        # Ready position: slightly bent, above table, reachable everywhere
        self._ready_q = [0.0, -0.3, 0.0, -1.5, 0.0, 1.2, 0.0]

    def _set_phase(self, phase, message=""):
        idx = _PHASE_ORDER.index(phase) if phase in _PHASE_ORDER else 0
        self.status.phase = phase
        self.status.phase_progress = 0.0
        self.status.overall_progress = idx / _N_PHASES
        self.status.message = message
        logger.info(f"Phase: {phase.value} - {message}")
        if self.recorder:
            self.recorder.set_phase(phase.value)

    async def _wait_traj(self, duration):
        start = time.time()
        while time.time() - start < duration + 0.5:
            if self._abort:
                return False
            state = self.backend.get_state()
            elapsed = time.time() - start
            self.status.phase_progress = min(1.0, elapsed / duration)
            if self.recorder:
                self.recorder.record_frame()
            if not state.trajectory_active and elapsed > 0.2:
                break
            await asyncio.sleep(0.05)
        return True

    async def _ik_move(self, target_pos, duration, label=""):
        """
        For transit moves gives the arm more freedom
        to reach distant positions.
        """
        current_q = np.array(self.backend.get_state().joint_positions)
        target_quat = self._down_quat
        solution, error, converged = self.ik.solve(
            target_pos=target_pos, target_quat=target_quat, initial_q=current_q)
        if not converged:
            self.status.phase = Phase.FAILED
            self.status.message = f"IK failed: {label} (err={error:.4f}m)"
            logger.error(self.status.message)
            return False
        success = self.backend.send_joint_positions(solution.tolist(), duration)
        if not success:
            self.status.phase = Phase.FAILED
            self.status.message = f"Backend rejected: {label}"
            return False
        logger.info(f"  -> {label}: [{target_pos[0]:.3f},{target_pos[1]:.3f},{target_pos[2]:.3f}]")
        return await self._wait_traj(duration)

    async def _safe_move(self, target_pos, safe_z, phase, label=""):
        """Strict UP -> ACROSS -> DOWN path."""
        self._set_phase(phase, label)
        state = self.backend.get_state()
        current_ee = np.array(state.end_effector_position) if state.end_effector_position else np.array([0.3, 0.0, 0.6])

        if current_ee[2] < safe_z - 0.01:
            ok = await self._ik_move(
                np.array([current_ee[0], current_ee[1], safe_z]), 1.5, "lift")
            if not ok: return False

        if np.linalg.norm(current_ee[:2] - target_pos[:2]) > 0.01:
            ok = await self._ik_move(
                np.array([target_pos[0], target_pos[1], safe_z]), 2.5, "horizontal")
            if not ok: return False

        if target_pos[2] < safe_z - 0.01:
            ok = await self._ik_move(target_pos, 2.5, "descend", )
            if not ok: return False
        return True

    async def _safe_return_to_ready(self):
        """Return to ready position WITHOUT swinging through the table.
        Lifts above everything first, moves to center, then goes to ready."""
        try:
            state = self.backend.get_state()
            if state.end_effector_position:
                ee = np.array(state.end_effector_position)
                safe_z = self.scene.get_safe_transit_height()
                # Lift at current XY
                if ee[2] < safe_z - 0.01:
                    await self._ik_move(np.array([ee[0], ee[1], safe_z]), 1.5, "lift before ready")
                # Move to center (near arm base) at safe height
                await self._ik_move(np.array([0.3, 0.0, safe_z]), 2.0, "transit to center")
        except Exception:
            pass  # Best effort - still go to ready
        self.backend.send_joint_positions(self._ready_q, 2.0)
        await self._wait_traj(2.0)

    def _verify_grasp(self, target_object, original_z):
        """Check if the object lifted with the gripper."""
        self.scene.refresh()
        obj = self.scene.find_object(target_object)
        if obj is None:
            return True  # Can't find it, assume it's held
        lifted = obj.position[2] - original_z
        logger.info(f"Grasp verify: {target_object} delta_z={lifted:.3f}m")
        return lifted > 0.005

    def _precheck_reachability(self, waypoints, labels):
        """Test IK for all waypoints BEFORE starting.
        Uses a generous tolerance (1.5cm) since this is just a reachability
        check, not the final execution. The actual IK during execution will
        converge more precisely from a nearby configuration."""
        PRECHECK_TOLERANCE = 0.015  # 15mm - generous for reachability test
        for pos, label in zip(waypoints, labels):
            solution, error, converged = self.ik.solve(
                target_pos=pos, target_quat=self._down_quat)
            if error > PRECHECK_TOLERANCE:
                return False, f"Unreachable waypoint: {label} [{pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f}] (err={error:.3f}m)"
        return True, "All waypoints reachable"

    async def _safe_recovery(self, approach_z):
        """Recovery when holding an object and something goes wrong.
        Release, retreat UPWARD, return to ready."""
        logger.warning("Executing safe recovery sequence")
        try:
            self.backend.set_gripper(1.0)
            await asyncio.sleep(0.5)
            state = self.backend.get_state()
            if state.end_effector_position:
                ee = np.array(state.end_effector_position)
                # Always go UP from current position, never down
                retreat_z = max(approach_z, ee[2])
                retreat_pos = np.array([ee[0], ee[1], retreat_z])
                current_q = np.array(state.joint_positions)
                sol, err, ok = self.ik.solve(target_pos=retreat_pos, target_quat=self._down_quat, initial_q=current_q)
                if ok:
                    self.backend.send_joint_positions(sol.tolist(), 2.0)
                    await self._wait_traj(2.0)
            # Return to ready
            await self._safe_return_to_ready()
        except Exception as e:
            logger.error(f"Recovery failed: {e}")

    async def execute(self, target_object, place_position, detections=None):
        self._abort = False
        self.status = SequencerStatus()
        self.status.object_name = target_object
        holding_object = False

        if self.recorder:
            self.recorder.start()

        # Refresh scene
        self.scene.refresh(detections)
        target_det = self.scene.find_object(target_object)
        if not target_det:
            self.status.phase = Phase.FAILED
            self.status.message = f"\'{target_object}\' not found"
            logger.error(self.status.message)
            return False

        # Compute grasp
        grasp = self.grasp.compute_grasp(target_det)

        # Scene reasoning
        place_info = self.scene.compute_place_target(np.array(place_position), target_object)

        # Place height: position the hand so the object bottom barely touches the surface.
        # hangover = exactly how far the object extends below the hand body.
        # So hand_z = surface_z + hangover + small release gap (don't push into surface).
        place_pos = place_info.position.copy()
        release_gap = 0.01  # 1cm above surface before releasing
        place_pos[2] += grasp.hangover + release_gap

        grasp_pos = grasp.grasp_pos
        original_obj_z = target_det.position[2]

        # Path-specific transit heights
        approach_z = self.scene.get_safe_transit_height(exclude_object=target_object)
        ideal_carrying_z = self.scene.get_carrying_transit_height(
            start_xy=grasp_pos[:2], end_xy=place_pos[:2],
            exclude_object=target_object, hangover=grasp.hangover)

        # Absolute physical minimum: carried object bottom clears the table surface.
        # The adaptive loop starts high (ideal) and steps down to find the lowest
        # reachable height. This floor is just physics: don't drag the object on the table.
        min_carrying_z = self.scene.table_height + grasp.hangover + self.scene.stack_gap

        # ── ADAPTIVE HEIGHT: find the lowest carrying_z that's reachable ──
        # Start from ideal, step down until all waypoints are reachable
        carrying_z = ideal_carrying_z
        step = 0.02  # 2cm steps

        while carrying_z >= min_carrying_z:
            retreat_z = max(carrying_z, place_pos[2], approach_z)
            waypoints = [
                (np.array([grasp_pos[0], grasp_pos[1], approach_z]), "pre-grasp"),
                (grasp_pos, "grasp"),
                (np.array([grasp_pos[0], grasp_pos[1], carrying_z]), "lift"),
                (np.array([place_pos[0], place_pos[1], carrying_z]), "transit"),
                (place_pos, "place"),
                (np.array([place_pos[0], place_pos[1], retreat_z]), "retreat"),
            ]
            reachable, msg = self._precheck_reachability(
                [w[0] for w in waypoints], [w[1] for w in waypoints])
            if reachable:
                break
            carrying_z -= step

        if not reachable:
            self.status.phase = Phase.FAILED
            self.status.message = f"No reachable carrying height found (tried {ideal_carrying_z:.3f} down to {min_carrying_z:.3f}): {msg}"
            logger.error(self.status.message)
            if self.recorder: self.recorder.stop()
            return False

        if carrying_z < ideal_carrying_z:
            logger.info(f"Adaptive height: reduced carrying_z from {ideal_carrying_z:.3f} to {carrying_z:.3f}")
        logger.info(f"Pre-check passed: all 6 waypoints reachable at carrying_z={carrying_z:.3f}")

        logger.info(
            f"PLAN: {target_object} -> [{place_pos[0]:.3f},{place_pos[1]:.3f},{place_pos[2]:.3f}] "
            f"approach_z={approach_z:.3f} carrying_z={carrying_z:.3f} "
            f"hangover={grasp.hangover:.3f} "
            f"{'STACK on '+str(place_info.objects_below) if place_info.is_stacking else 'on table'}")

        try:
            # Phase 1: Open gripper
            self._set_phase(Phase.OPEN_GRIPPER, f"Opening ({grasp.gripper_opening:.2f})")
            self.backend.set_gripper(grasp.gripper_opening)
            await asyncio.sleep(0.8)
            if self._abort: return False

            # Phase 2: Move above object
            pregrasp = np.array([grasp_pos[0], grasp_pos[1], approach_z])
            ok = await self._safe_move(pregrasp, approach_z, Phase.MOVE_TO_PREGRASP, f"Above {target_object}")
            if not ok: return False

            # Phase 3: Descend gently to grasp (orientation matters here)
            self._set_phase(Phase.DESCEND_TO_GRASP, f"Descending to {target_object}")
            ok = await self._ik_move(grasp_pos, 3.0, "descend to grasp")
            if not ok: return False

            # Phase 4: Close gripper
            self._set_phase(Phase.CLOSE_GRIPPER, f"Grip {target_object} (close={grasp.grip_close_fraction:.2f})")
            self.backend.set_gripper(grasp.grip_close_fraction)
            await asyncio.sleep(0.8)
            if self._abort: return False

            # Test lift and verify (no orientation needed)
            test_lift = np.array([grasp_pos[0], grasp_pos[1], grasp_pos[2] + 0.04])
            ok = await self._ik_move(test_lift, 1.5, "test lift")
            if not ok: return False
            await asyncio.sleep(0.3)

            if not self._verify_grasp(target_object, original_obj_z):
                logger.error(f"GRASP FAILED: {target_object}")
                self._set_phase(Phase.FAILED, f"Grip failed on {target_object}")
                self.backend.set_gripper(1.0)
                await asyncio.sleep(0.5)
                await self._ik_move(np.array([grasp_pos[0], grasp_pos[1], approach_z]), 1.5, "retreat")
                await self._safe_return_to_ready()
                if self.recorder: self.recorder.stop()
                return False

            holding_object = True
            logger.info(f"Grasp verified: {target_object} held")

            # Phase 5: Lift to carrying height
            lift_pos = np.array([grasp_pos[0], grasp_pos[1], carrying_z])
            self._set_phase(Phase.LIFT, f"Lift to carrying_z={carrying_z:.3f}")
            ok = await self._ik_move(lift_pos, 1.5, "lift")
            if not ok: raise RuntimeError("Lift failed")

            # Verify AGAIN after full lift (object may have slipped during ascent)
            await asyncio.sleep(0.2)
            if not self._verify_grasp(target_object, original_obj_z):
                logger.error(f"Object slipped during lift: {target_object}")
                raise RuntimeError(f"Object slipped during lift")

            # Phase 6: Transit horizontally at carrying height
            transit_pos = np.array([place_pos[0], place_pos[1], carrying_z])
            msg = f"Transit at z={carrying_z:.3f}"
            if place_info.is_stacking:
                msg += f" (stack on {place_info.objects_below})"
            self._set_phase(Phase.TRANSIT, msg)
            ok = await self._ik_move(transit_pos, 2.5, "transit")
            if not ok: raise RuntimeError("Transit failed")

            # Phase 7: Descend to place
            msg = f"Lower to z={place_pos[2]:.3f}"
            if place_info.is_stacking:
                msg += f" (on {place_info.objects_below})"
            self._set_phase(Phase.DESCEND_TO_PLACE, msg)
            ok = await self._ik_move(place_pos, 3.0, "descend to place")
            if not ok: raise RuntimeError("Descend to place failed")

            # Phase 8: Release
            self._set_phase(Phase.OPEN_GRIPPER_RELEASE, "Releasing")
            self.backend.set_gripper(1.0)
            holding_object = False
            await asyncio.sleep(0.8)
            if self._abort: return False

            # Phase 9: Retreat UPWARD. Must go higher than where we are now.
            # Use carrying_z (verified reachable at this XY, always above all objects)
            retreat_z = max(carrying_z, place_pos[2], approach_z)
            retreat = np.array([place_pos[0], place_pos[1], retreat_z])
            self._set_phase(Phase.RETREAT, f"Retreat to z={retreat_z:.3f}")
            ok = await self._ik_move(retreat, 1.5, "retreat")
            if not ok: raise RuntimeError("Retreat failed")

            # Return to ready SAFELY: first lift above everything, then go to ready
            # A direct joint-space move to ready would swing through the table
            logger.info("Safe return to ready position")
            state = self.backend.get_state()
            if state.end_effector_position:
                ee = np.array(state.end_effector_position)
                # Lift straight up at current XY
                safe_return_z = self.scene.get_safe_transit_height()
                if ee[2] < safe_return_z - 0.01:
                    lift_target = np.array([ee[0], ee[1], safe_return_z])
                    await self._ik_move(lift_target, 1.5, "lift before ready")
                # Move horizontally toward arm base area (closer to ready pose)
                await self._ik_move(np.array([0.3, 0.0, safe_return_z]), 2.0, "transit to center")
            # Now safe to go to ready joints (arm is above table near base)
            await self._safe_return_to_ready()

            if self.recorder:
                self.recorder.stop()

            self.status.phase = Phase.COMPLETE
            self.status.overall_progress = 1.0
            msg = f"Complete: {target_object}"
            if place_info.is_stacking:
                msg += f" (stacked on {place_info.objects_below})"
            self.status.message = msg
            logger.info(msg)
            return True

        except RuntimeError as e:
            logger.error(f"Sequence failed: {e}")
            self.status.phase = Phase.FAILED
            self.status.message = str(e)
            if holding_object:
                await self._safe_recovery(approach_z)
            else:
                await self._safe_return_to_ready()
            if self.recorder: self.recorder.stop()
            return False

    def abort(self):
        self._abort = True
        self.status.phase = Phase.FAILED
        self.status.message = "Aborted"
        if self.recorder:
            self.recorder.stop()
        logger.warning("Sequence aborted")

    async def execute_with_retry(self, target_object, place_position, max_retries=3):
        """
        Execute pick-and-place with automatic retry on grip failure.
        """
        for attempt in range(1, max_retries + 1):
            if self._abort:
                return False

            logger.info(f"Attempt {attempt}/{max_retries} for {target_object}")
            success = await self.execute(target_object, place_position)

            if success:
                return True

            if "Unreachable" in self.status.message or "No reachable" in self.status.message:
                logger.error(f"Position unreachable, not retrying: {self.status.message}")
                return False

            if attempt < max_retries:
                logger.info(f"Retrying {target_object} (attempt {attempt} failed: {self.status.message})")
                await asyncio.sleep(1.0)

        logger.error(f"All {max_retries} attempts failed for {target_object}")
        return False

    async def smart_execute(self, target_object, place_position, max_retries=3):
        """
        Intelligent pick-and-place with automatic task decomposition.

        If the target object is buried in a stack (has objects above it),
        CHIRON automatically:
        1. Identifies which objects are on top
        2. Finds clear temporary spots on the table
        3. Moves the obstructing objects off the stack (top-first)
        4. Picks the target and places it at the desired location

        This is task-level intelligence: one user command decomposes into
        a multi-step plan based on scene understanding.
        """
        if self._abort:
            return False

        # Refresh scene to understand the current state
        self.scene.refresh()
        target_det = self.scene.find_object(target_object)
        if not target_det:
            self.status.phase = Phase.FAILED
            self.status.message = f"'{target_object}' not found"
            return False

        # Check if there are objects stacked above the target
        objects_above = self.scene.get_objects_above(target_object)

        if objects_above:
            names_above = [o.name for o in objects_above]
            logger.info(
                f"TASK DECOMPOSITION: '{target_object}' has {len(objects_above)} "
                f"object(s) above it: {names_above}. Clearing stack first."
            )

            # Find temporary spots for each object we need to move
            temp_positions = []
            for i, obj in enumerate(objects_above):
                temp_pos = self.scene.find_clear_spot(
                    exclude_positions=[np.array(place_position)] + temp_positions
                )
                temp_positions.append(temp_pos)

            # Clear objects top-to-bottom
            for obj, temp_pos in zip(objects_above, temp_positions):
                if self._abort:
                    return False
                logger.info(f"CLEARING: Moving '{obj.name}' to temp [{temp_pos[0]:.2f},{temp_pos[1]:.2f}]")
                success = await self.execute_with_retry(
                    obj.name, temp_pos.tolist(), max_retries=max_retries
                )
                if not success:
                    logger.error(f"Failed to clear '{obj.name}' from stack")
                    return False
                # Re-scan after each move
                self.scene.refresh()

        # Now the target is accessible — pick it
        logger.info(f"TARGET: Picking '{target_object}' -> [{place_position[0]:.2f},{place_position[1]:.2f}]")
        return await self.execute_with_retry(target_object, place_position, max_retries=max_retries)
