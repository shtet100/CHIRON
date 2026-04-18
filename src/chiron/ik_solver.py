"""
CHIRON IK solver.

Damped least-squares Jacobian IK for MuJoCo models.
Converts Cartesian end-effector goals (position + orientation)
into joint positions that achieve those goals.

This is what lets ORION say "move to [x,y,z]" instead of
specifying raw joint angles.
"""

import mujoco
import numpy as np
import logging
from typing import Optional, Tuple

logger = logging.getLogger("chiron.ik")


class IKSolver:
    """
    Iterative IK using damped least-squares on the MuJoCo Jacobian.

    Works on any MuJoCo model — reads the Jacobian from mj_jac,
    computes joint deltas, and iterates until convergence.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        ee_body_name: str = "hand",
        arm_joint_ids: list = None,
        max_iterations: int = 1000,
        position_tolerance: float = 0.005,   # 5mm
        orientation_tolerance: float = 0.03,  # ~1.7 degrees
        damping: float = 1e-6,
    ):
        self.model = model
        self._live_data = data  # Reference to live data (read-only for seeding)
        # Own copy of MjData for IK solving — avoids threading collisions
        # with the sim loop which steps the live data at 500 Hz
        self._ik_data = mujoco.MjData(model)
        self.max_iterations = max_iterations
        self.pos_tol = position_tolerance
        self.ori_tol = orientation_tolerance
        self.damping = damping

        # A good initial config for reaching tasks (arm slightly bent forward)
        # Avoids the near-singular home configuration
        self.default_init_q = [0.0, -0.3, 0.0, -1.5, 0.0, 1.2, 0.0]

        # Resolve EE body
        self.ee_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ee_body_name)
        if self.ee_body_id < 0:
            raise ValueError(f"Body '{ee_body_name}' not found in model")

        # Arm joint indices (exclude freejoints from objects, finger joints)
        if arm_joint_ids is not None:
            self.arm_joint_ids = arm_joint_ids
        else:
            self.arm_joint_ids = []
            for i in range(model.njnt):
                name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) or ""
                jnt_type = model.jnt_type[i]
                # Only hinge joints that aren't fingers/grippers
                if jnt_type == mujoco.mjtJoint.mjJNT_HINGE:
                    if not name.startswith("finger") and not name.startswith("gripper"):
                        self.arm_joint_ids.append(i)

        self.n_joints = len(self.arm_joint_ids)
        logger.info(f"IK solver initialized: ee_body={ee_body_name}, {self.n_joints} arm joints")

    def solve(
        self,
        target_pos: np.ndarray,
        target_quat: Optional[np.ndarray] = None,
        initial_q: Optional[np.ndarray] = None,
    ) -> Tuple[Optional[np.ndarray], float, bool]:
        """
        Solve IK for a target end-effector pose.

        Args:
            target_pos: [x, y, z] target position in world frame
            target_quat: [w, x, y, z] target orientation (MuJoCo convention).
                         If None, only position is controlled.
            initial_q: Starting joint configuration. If None, uses a
                       pre-set reaching pose that avoids singularities.

        Returns:
            (joint_positions, final_error, converged)
            joint_positions is None if IK failed completely.
        """
        target_pos = np.array(target_pos, dtype=np.float64)
        use_orientation = target_quat is not None
        if use_orientation:
            target_quat = np.array(target_quat, dtype=np.float64)
            target_quat = target_quat / np.linalg.norm(target_quat)

        # Work on our private MjData copy (thread-safe, no save/restore needed)
        data = self._ik_data

        # Copy current live state as baseline (preserves object positions, etc.)
        data.qpos[:] = self._live_data.qpos.copy()
        data.qvel[:] = 0
        data.ctrl[:] = 0

        # Set initial arm configuration
        init = initial_q if initial_q is not None else self.default_init_q
        for i, jid in enumerate(self.arm_joint_ids):
            if i < len(init):
                data.qpos[jid] = init[i]

        # Iterative IK
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))

        # Map joint ids to velocity (dof) indices
        dof_ids = [self.model.jnt_dofadr[jid] for jid in self.arm_joint_ids]

        converged = False
        final_error = float('inf')

        for iteration in range(self.max_iterations):
            mujoco.mj_forward(self.model, data)

            # Current EE pose
            current_pos = data.xpos[self.ee_body_id].copy()
            current_quat = data.xquat[self.ee_body_id].copy()

            # Position error
            pos_error = target_pos - current_pos

            # Orientation error (as axis-angle)
            ori_error = np.zeros(3)
            if use_orientation:
                current_quat_inv = current_quat.copy()
                current_quat_inv[1:] *= -1
                error_quat = np.zeros(4)
                mujoco.mju_mulQuat(error_quat, target_quat, current_quat_inv)
                if error_quat[0] < 0:
                    error_quat *= -1
                ori_error = 2.0 * error_quat[1:]

            # Check convergence
            pos_mag = np.linalg.norm(pos_error)
            ori_mag = np.linalg.norm(ori_error)
            final_error = pos_mag

            if pos_mag < self.pos_tol and (not use_orientation or ori_mag < self.ori_tol):
                converged = True
                break

            # Compute Jacobian at current EE position
            mujoco.mj_jac(self.model, data, jacp, jacr, current_pos, self.ee_body_id)

            # Extract columns for our arm joints only
            J_pos = jacp[:, dof_ids]

            if use_orientation:
                J_ori = jacr[:, dof_ids]
                J = np.vstack([J_pos, J_ori])
                error = np.concatenate([pos_error, ori_error])
            else:
                J = J_pos
                error = pos_error

            # Damped least-squares: dq = J^T (J J^T + lambda I)^-1 error
            JJT = J @ J.T
            n = JJT.shape[0]
            dq = J.T @ np.linalg.solve(JJT + self.damping * np.eye(n), error)

            # Clip step size for stability
            dq = np.clip(dq, -0.1, 0.1)

            # Update joint positions with clamping to limits
            for i, jid in enumerate(self.arm_joint_ids):
                data.qpos[jid] += dq[i]
                lo, hi = self.model.jnt_range[jid]
                if hi > lo:
                    data.qpos[jid] = np.clip(data.qpos[jid], lo, hi)

        # Extract solution
        solution = np.array([data.qpos[jid] for jid in self.arm_joint_ids])

        if converged:
            logger.debug(f"IK converged in {iteration+1} iterations, error={final_error:.4f}m")
        else:
            # Retry with default init config if we used a custom one
            # (the provided initial_q may be near a singularity)
            if initial_q is not None:
                logger.debug(f"IK retry with default init (first attempt error={final_error:.4f}m)")
                return self.solve(target_pos, target_quat, initial_q=None)
            logger.warning(f"IK did not converge after {self.max_iterations} iters, error={final_error:.4f}m")

        return solution, final_error, converged

    def get_ee_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return current (position, quat_wxyz) of the end-effector from live data."""
        mujoco.mj_forward(self.model, self._live_data)
        pos = self._live_data.xpos[self.ee_body_id].copy()
        quat = self._live_data.xquat[self.ee_body_id].copy()
        return pos, quat
