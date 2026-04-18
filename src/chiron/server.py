"""
CHIRON FastAPI server.

Upstream face of the bridge: exposes WebSocket + REST endpoints
that ORION (port 8000) connects to for robot control and state streaming.

Endpoints:
    GET  /            - Live dashboard UI
    GET  /health      - Backend status
    GET  /robot_info  - Joint names, limits, DOF
    POST /e_stop      - Emergency stop
    WS   /ws/state    - Robot state stream (CHIRON -> ORION)
    WS   /ws/command  - Command receiver (ORION -> CHIRON)
"""

import asyncio
import json
import time
import logging
from typing import Optional, Set

from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, HTMLResponse

from .backends.base import RobotBackend
from .sequencer import PickAndPlaceSequencer, Phase, SequencerStatus
from .trajectory_recorder import TrajectoryRecorder

_DASHBOARD_HTML = (Path(__file__).parent / "dashboard.html").read_text()

logger = logging.getLogger("chiron.server")

app = FastAPI(title="CHIRON", version="0.3.0", description="ORION Motor Cortex")

# ── Global state (set by main.py before server starts) ──────────────
backend: Optional[RobotBackend] = None
sequencer: Optional[PickAndPlaceSequencer] = None
recorder: Optional[TrajectoryRecorder] = None
state_rate_hz: int = 50
_active_task: Optional[asyncio.Task] = None

# Track active WebSocket connections for clean shutdown
_state_clients: Set[WebSocket] = set()


# ── REST endpoints ──────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    return _DASHBOARD_HTML


@app.get("/health")
async def health():
    if backend is None:
        return JSONResponse(
            status_code=503,
            content={"status": "no_backend", "backend": None}
        )
    info = backend.get_info()
    return {
        "status": "ok",
        "backend": type(backend).__name__,
        "robot": info.name,
        "dof": info.dof,
        "state_clients": len(_state_clients),
    }


@app.get("/robot_info")
async def robot_info():
    if backend is None:
        return JSONResponse(status_code=503, content={"error": "no backend connected"})
    info = backend.get_info()
    return {
        "name": info.name,
        "dof": info.dof,
        "joint_names": info.joint_names,
        "joint_limits_lower": info.joint_limits_lower,
        "joint_limits_upper": info.joint_limits_upper,
        "has_gripper": info.has_gripper,
    }


@app.post("/e_stop")
async def e_stop():
    if backend is None:
        return JSONResponse(status_code=503, content={"error": "no backend connected"})
    if sequencer:
        sequencer.abort()
    success = backend.emergency_stop()
    return {"success": success}


@app.get("/scene_objects")
async def scene_objects():
    if backend is None:
        return JSONResponse(status_code=503, content={"error": "no backend connected"})
    from .grasp_computer import scan_scene_objects
    detections = scan_scene_objects(backend.model, backend.data)
    return {
        "objects": [
            {
                "name": d.name, "shape": d.shape,
                "position": d.position.tolist(), "dimensions": d.dimensions,
            }
            for d in detections
        ]
    }


@app.get("/sequencer_status")
async def sequencer_status():
    if sequencer is None:
        return {"phase": "no_sequencer"}
    return sequencer.status.to_dict()


@app.get("/scene")
async def scene_summary():
    if sequencer is None or sequencer.scene is None:
        return {"error": "no scene awareness"}
    sequencer.scene.refresh()
    return sequencer.scene.get_scene_summary()


@app.post("/reset")
async def reset_scene():
    """Reset the simulation to initial state."""
    global _active_task
    if backend is None:
        return JSONResponse(status_code=503, content={"error": "no backend"})

    # Cancel any running pick-and-place task
    if sequencer:
        sequencer.abort()
    if _active_task and not _active_task.done():
        _active_task.cancel()
        try:
            await _active_task
        except (asyncio.CancelledError, Exception):
            pass
        _active_task = None

    # Reset physics
    import mujoco
    mujoco.mj_resetData(backend.model, backend.data)
    mujoco.mj_forward(backend.model, backend.data)

    # Clean up sequencer state so new commands work immediately
    if sequencer:
        sequencer._abort = False
        sequencer.status = SequencerStatus()  # Fresh status: IDLE
    if recorder and recorder.is_recording:
        recorder.stop()

    logger.info("Scene reset to initial state")
    return {"success": True, "message": "Scene reset"}


@app.post("/move_target")
async def move_target(x: float = 0.4, y: float = -0.3):
    """Move the green target zone to a new position."""
    if backend is None:
        return JSONResponse(status_code=503, content={"error": "no backend"})
    import mujoco
    if backend.model.nmocap > 0:
        backend.data.mocap_pos[0][0] = x
        backend.data.mocap_pos[0][1] = y
        backend.data.mocap_pos[0][2] = 0.401
        return {"success": True, "position": [x, y, 0.401]}
    return {"success": False, "message": "No mocap body found"}


@app.get("/trajectory")
async def trajectory_data():
    """Get recorded trajectory data for DAEDALUS."""
    if recorder is None:
        return {"error": "no recorder"}
    return recorder.summary()


# ── WebSocket: state stream (CHIRON -> ORION) ──────────────────────

@app.websocket("/ws/state")
async def state_stream(ws: WebSocket):
    await ws.accept()
    _state_clients.add(ws)
    logger.info(f"State client connected ({len(_state_clients)} total)")

    try:
        interval = 1.0 / state_rate_hz
        while True:
            if backend is None:
                await asyncio.sleep(0.5)
                continue

            state = backend.get_state()
            msg_data = state.to_dict()
            # Include sequencer status if active
            if sequencer is not None:
                msg_data["sequencer"] = sequencer.status.to_dict()
            await ws.send_json({"type": "state", "data": msg_data})
            await asyncio.sleep(interval)

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"State stream error: {e}")
    finally:
        _state_clients.discard(ws)
        logger.info(f"State client disconnected ({len(_state_clients)} remaining)")


# ── WebSocket: command receiver (ORION -> CHIRON) ──────────────────

@app.websocket("/ws/command")
async def command_receiver(ws: WebSocket):
    await ws.accept()
    logger.info("Command client connected")

    try:
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await ws.send_json({"type": "error", "message": "Invalid JSON"})
                continue

            cmd_type = msg.get("type")
            data = msg.get("data", {})

            if backend is None:
                await ws.send_json({"type": "error", "message": "No backend connected"})
                continue

            response = await _handle_command(cmd_type, data)
            await ws.send_json(response)

    except WebSocketDisconnect:
        logger.info("Command client disconnected")
    except Exception as e:
        logger.error(f"Command receiver error: {e}")


async def _handle_command(cmd_type: str, data: dict) -> dict:
    """Dispatch a command to the active backend. Returns a response dict."""

    if cmd_type == "move_joints":
        positions = data.get("positions", [])
        duration = data.get("duration", 2.0)
        success = backend.send_joint_positions(positions, duration)
        return {"type": "ack", "command": cmd_type, "success": success}

    elif cmd_type == "gripper":
        state = data.get("state", "open")
        if isinstance(state, str):
            openness = 1.0 if state == "open" else 0.0 if state == "close" else 0.5
        else:
            openness = float(state)
        success = backend.set_gripper(openness)
        return {"type": "ack", "command": cmd_type, "success": success}

    elif cmd_type == "emergency_stop":
        success = backend.emergency_stop()
        return {"type": "ack", "command": cmd_type, "success": success}

    elif cmd_type == "move_cartesian":
        if not hasattr(backend, 'model'):
            return {"type": "error", "message": "IK not available for this backend"}
        from .ik_solver import IKSolver
        ik = IKSolver(backend.model, backend.data, arm_joint_ids=backend._arm_joint_ids)
        target_pos = data.get("position", [0.4, 0.0, 0.5])
        target_quat = data.get("orientation", None)  # [w,x,y,z] or null
        duration = data.get("duration", 2.0)
        solution, error, converged = ik.solve(target_pos=target_pos, target_quat=target_quat)
        if not converged:
            return {"type": "error", "message": f"IK failed (error={error:.4f}m)"}
        success = backend.send_joint_positions(solution.tolist(), duration)
        return {"type": "ack", "command": cmd_type, "success": success, "ik_error": round(error, 5)}

    elif cmd_type == "pick_place":
        if sequencer is None:
            return {"type": "error", "message": "Sequencer not initialized (need workspace scene)"}
        target_object = data.get("object", "red_cube")
        place_pos = data.get("place_position", [0.4, -0.3, 0.40])
        # Run sequence in background so we can ACK immediately
        global _active_task
        _active_task = asyncio.create_task(_run_pick_place(target_object, place_pos))
        return {"type": "ack", "command": cmd_type, "success": True,
                "message": f"Pick-and-place started: {target_object}"}

    elif cmd_type == "abort_sequence":
        if sequencer:
            sequencer.abort()
            backend.emergency_stop()
        return {"type": "ack", "command": cmd_type, "success": True}

    else:
        return {"type": "error", "message": f"Unknown command: {cmd_type}"}


async def _run_pick_place(target_object: str, place_position: list):
    """Run pick-and-place in the background."""
    global _active_task
    try:
        success = await sequencer.smart_execute(target_object, place_position, max_retries=3)
        if success:
            logger.info(f"Pick-and-place complete: {target_object}")
        else:
            logger.error(f"Pick-and-place failed: {sequencer.status.message}")
    except asyncio.CancelledError:
        logger.info("Pick-and-place cancelled by reset")
    except Exception as e:
        logger.error(f"Pick-and-place error: {e}")
    finally:
        _active_task = None
