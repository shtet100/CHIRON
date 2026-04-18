"""
CHIRON entry point.

Initializes the configured backend, starts the simulation loop
in a background thread, and runs the FastAPI server.

Usage:
    conda activate chiron
    cd ~/CHIRON
    python main.py                  # headless (dashboard only)
    python main.py --render         # with 3D MuJoCo viewer
    python main.py --config config/chiron_config.yaml
"""

import argparse
import logging
import sys
import threading
import time
from pathlib import Path

import uvicorn

from src.chiron.config import load_config
from src.chiron.backends.mujoco_backend import MuJoCoBackend
from src.chiron.ik_solver import IKSolver
from src.chiron.gripper_model import measure_gripper
from src.chiron.grasp_computer import GraspComputer
from src.chiron.scene_awareness import SceneAwareness
from src.chiron.trajectory_recorder import TrajectoryRecorder
from src.chiron.sequencer import PickAndPlaceSequencer
import src.chiron.server as server

# ── Logging setup ──────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("chiron.main")


# ── Simulation loop ────────────────────────────────────────────────

def sim_loop(backend: MuJoCoBackend, dt: float, viewer=None):
    """
    Step the simulation in a background thread.
    Runs at ~500 Hz (dt=0.002). The Lock inside the backend
    ensures thread safety with the server's state reads.
    If a viewer is provided, syncs the 3D render after each step.
    """
    logger.info(f"Sim loop started (dt={dt}s, ~{int(1/dt)} Hz, render={'on' if viewer else 'off'})")

    while True:
        backend.step()

        # Sync the 3D viewer if active
        if viewer is not None:
            if not viewer.is_running():
                logger.info("Viewer window closed")
                viewer = None  # Stop trying to sync
            else:
                viewer.sync()

        time.sleep(dt)


# ── Backend factory ────────────────────────────────────────────────

def create_backend(config):
    """Instantiate the backend specified in config."""
    name = config.active_backend

    if name == "mujoco":
        model_path = Path(config.mujoco.model_file)
        if not model_path.exists():
            logger.error(f"MuJoCo model not found: {model_path}")
            sys.exit(1)
        return MuJoCoBackend(str(model_path), dt=config.mujoco.dt)

    elif name == "genesis":
        logger.error("Genesis backend not yet implemented (Phase C1)")
        sys.exit(1)

    elif name == "lerobot":
        logger.error("LeRobot backend not yet implemented (Phase C4)")
        sys.exit(1)

    else:
        logger.error(f"Unknown backend: {name}")
        sys.exit(1)


# ── Main ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CHIRON - ORION Motor Cortex")
    parser.add_argument(
        "--config", default="config/chiron_config.yaml",
        help="Path to chiron_config.yaml"
    )
    parser.add_argument(
        "--render", action="store_true",
        help="Launch MuJoCo 3D viewer window alongside the server"
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Create and connect backend
    backend = create_backend(config)
    logger.info(f"Connecting to {config.active_backend} backend...")

    if not backend.connect():
        logger.error("Backend connection failed")
        sys.exit(1)

    info = backend.get_info()
    logger.info(f"Connected: {info.name} ({info.dof} DOF, joints: {info.joint_names})")

    # Wire backend into the server module
    server.backend = backend
    server.state_rate_hz = config.server.state_publish_rate_hz

    # Initialize IK solver and pick-and-place sequencer (MuJoCo only)
    if config.active_backend == "mujoco":
        try:
            # IK solver
            ik_solver = IKSolver(
                backend.model, backend.data,
                ee_body_name="hand",
                arm_joint_ids=backend._arm_joint_ids,
            )

            # Measure gripper from model (zero hardcoded values)
            gripper_geom = measure_gripper(backend.model, backend.data)

            # Grasp computer using measured gripper
            grasp_computer = GraspComputer(gripper_geom)

            # Scene awareness (gripper_clearance from measured geometry)
            scene = SceneAwareness(
                model=backend.model,
                data=backend.data,
                table_height=0.40,
                clearance_margin=0.03,
                gripper_clearance=gripper_geom.pad_bottom_offset,
            )

            # Trajectory recorder for DAEDALUS
            recorder = TrajectoryRecorder(backend)

            # Sequencer
            seq = PickAndPlaceSequencer(
                backend=backend,
                ik_solver=ik_solver,
                grasp_computer=grasp_computer,
                scene=scene,
                gripper_geom=gripper_geom,
                recorder=recorder,
            )
            server.sequencer = seq
            server.recorder = recorder
            logger.info("Pick-and-place sequencer initialized (all geometry computed from model)")
        except Exception as e:
            logger.warning(f"Could not initialize sequencer: {e}")
            import traceback; traceback.print_exc()
            logger.warning("Pick-and-place commands will not be available")

    # Launch 3D viewer if requested
    viewer = None
    if args.render and config.active_backend == "mujoco":
        try:
            import mujoco.viewer
            logger.info("Launching MuJoCo 3D viewer...")
            viewer = mujoco.viewer.launch_passive(backend.model, backend.data)
            logger.info("MuJoCo viewer opened. Use dashboard + viewer side by side.")
        except Exception as e:
            logger.warning(f"Could not launch viewer: {e}")
            logger.warning("Continuing without 3D viewer (dashboard still available)")

    # Start sim loop in background thread
    if config.active_backend in ("mujoco", "genesis"):
        dt = config.mujoco.dt if config.active_backend == "mujoco" else config.genesis.dt
        sim_thread = threading.Thread(target=sim_loop, args=(backend, dt, viewer), daemon=True)
        sim_thread.start()

    # Start FastAPI server
    logger.info(f"CHIRON listening on {config.server.host}:{config.server.port}")
    uvicorn.run(
        server.app,
        host=config.server.host,
        port=config.server.port,
        log_level="warning",  # uvicorn's own logs; we use our logger
    )


if __name__ == "__main__":
    main()
