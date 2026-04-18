"""
CHIRON Phase C0 integration test.

Tests the full vertical slice: backend -> server -> WebSocket -> client.
Run with: python tests/test_phase_c0.py

Expects CHIRON server to NOT be running (this script starts its own).
"""

import asyncio
import json
import sys
import threading
import time

# Add project root to path
sys.path.insert(0, ".")

from src.chiron.backends.mujoco_backend import MuJoCoBackend
from src.chiron.config import load_config


def test_backend_standalone():
    """Test the MuJoCo backend without the server."""
    print("=" * 60)
    print("TEST 1: Backend standalone")
    print("=" * 60)

    backend = MuJoCoBackend("models/mujoco_menagerie/franka_emika_panda/scene.xml")
    assert backend.connect(), "FAIL: connect()"

    info = backend.get_info()
    assert info.dof == 7, f"FAIL: expected 7 DOF, got {info.dof}"
    assert info.has_gripper, "FAIL: expected gripper"
    assert len(info.joint_names) == 7, f"FAIL: expected 7 joint names"
    print(f"  Robot: {info.name}, {info.dof} DOF, gripper={info.has_gripper}")
    print(f"  Joints: {info.joint_names}")

    # Step sim and read state
    for _ in range(200):
        backend.step()

    state = backend.get_state()
    assert len(state.joint_positions) == 7, "FAIL: wrong position count"
    assert len(state.joint_velocities) == 7, "FAIL: wrong velocity count"
    assert len(state.joint_efforts) == 7, "FAIL: wrong effort count"
    assert state.end_effector_position is not None, "FAIL: no EE position"
    assert state.end_effector_orientation is not None, "FAIL: no EE orientation"
    print(f"  State OK: positions={[round(p,3) for p in state.joint_positions]}")
    print(f"  EE pos: {[round(x,4) for x in state.end_effector_position]}")

    # Command arm
    target = [0.0, -0.5, 0.5, -1.0, 0.0, 1.0, 0.0]
    assert backend.send_joint_positions(target, 2.0), "FAIL: send_joint_positions"
    for _ in range(1000):
        backend.step()
    state2 = backend.get_state()
    for i, (actual, expected) in enumerate(zip(state2.joint_positions, target)):
        assert abs(actual - expected) < 0.1, f"FAIL: joint {i} did not reach target ({actual:.3f} vs {expected:.3f})"
    print(f"  Joint command OK: reached target within tolerance")

    # E-stop
    assert backend.emergency_stop(), "FAIL: e-stop"
    print(f"  E-stop OK")

    # Gripper
    assert backend.set_gripper(1.0), "FAIL: set_gripper open"
    assert backend.set_gripper(0.0), "FAIL: set_gripper close"
    print(f"  Gripper OK")

    # Joint limit validation
    bad_target = [99.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0]
    assert not backend.send_joint_positions(bad_target, 2.0), "FAIL: should reject out-of-limit"
    print(f"  Joint limit validation OK")

    backend.disconnect()
    print("  PASSED\n")


def test_config_loading():
    """Test config loads from YAML and falls back to defaults."""
    print("=" * 60)
    print("TEST 2: Config loading")
    print("=" * 60)

    config = load_config("config/chiron_config.yaml")
    assert config.server.port == 8200
    assert config.active_backend == "mujoco"
    assert config.mujoco.dt == 0.002
    print(f"  Config loaded: port={config.server.port}, backend={config.active_backend}")

    # Test fallback to defaults
    config2 = load_config("nonexistent.yaml")
    assert config2.server.port == 8200
    assert config2.active_backend == "mujoco"
    print(f"  Fallback defaults OK")

    print("  PASSED\n")


async def test_server_websocket():
    """Test the full server with WebSocket connections."""
    import websockets

    print("=" * 60)
    print("TEST 3: Server + WebSocket")
    print("=" * 60)

    # Start server in background
    import uvicorn
    import src.chiron.server as server
    from src.chiron.backends.mujoco_backend import MuJoCoBackend

    backend = MuJoCoBackend("models/mujoco_menagerie/franka_emika_panda/scene.xml")
    assert backend.connect()
    server.backend = backend
    server.state_rate_hz = 50

    # Sim loop
    def sim_loop():
        while not stop_event.is_set():
            backend.step()
            time.sleep(0.002)

    stop_event = threading.Event()
    sim_thread = threading.Thread(target=sim_loop, daemon=True)
    sim_thread.start()

    # Server in background thread
    config = uvicorn.Config(server.app, host="127.0.0.1", port=8201, log_level="error")
    srv = uvicorn.Server(config)
    srv_thread = threading.Thread(target=srv.run, daemon=True)
    srv_thread.start()
    await asyncio.sleep(1.5)

    try:
        # Test state stream
        async with websockets.connect("ws://127.0.0.1:8201/ws/state") as ws:
            for i in range(3):
                msg = json.loads(await ws.recv())
                assert msg["type"] == "state"
                assert "joint_positions" in msg["data"]
                assert "end_effector_pose" in msg["data"]
            print(f"  State stream OK (3 frames received)")

        # Test commands
        async with websockets.connect("ws://127.0.0.1:8201/ws/command") as ws:
            # move_joints
            await ws.send(json.dumps({
                "type": "move_joints",
                "data": {"positions": [0.1, -0.2, 0.3, -0.5, 0.1, 0.5, -0.1], "duration": 1.0}
            }))
            ack = json.loads(await ws.recv())
            assert ack["type"] == "ack" and ack["success"], f"FAIL: move_joints ack: {ack}"
            print(f"  move_joints OK")

            # gripper
            await ws.send(json.dumps({"type": "gripper", "data": {"state": "open"}}))
            ack = json.loads(await ws.recv())
            assert ack["type"] == "ack" and ack["success"], f"FAIL: gripper ack: {ack}"
            print(f"  gripper OK")

            # e-stop
            await ws.send(json.dumps({"type": "emergency_stop"}))
            ack = json.loads(await ws.recv())
            assert ack["type"] == "ack" and ack["success"], f"FAIL: e-stop ack: {ack}"
            print(f"  e-stop OK")

            # unknown command
            await ws.send(json.dumps({"type": "fly_to_moon"}))
            err = json.loads(await ws.recv())
            assert err["type"] == "error", f"FAIL: expected error for unknown cmd: {err}"
            print(f"  unknown command rejection OK")

        print("  PASSED\n")

    finally:
        stop_event.set()
        srv.should_exit = True
        backend.disconnect()


def main():
    print("\nCHIRON Phase C0 Integration Tests")
    print("=" * 60)
    print()

    test_backend_standalone()
    test_config_loading()
    asyncio.run(test_server_websocket())

    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
