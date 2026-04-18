# CHIRON

**Cybernetic Hardware Interface for Robotic Operations & Networking**

---

CHIRON is a robot motor cortex that translates high-level commands into physically intelligent motion. It measures its own gripper, reasons about object geometry, plans collision-free paths, and executes pick-and-place sequences with automatic failure recovery. It serves as the hardware abstraction layer of [ORION](https://github.com/shtet100), a modular robotics ecosystem.

The entire system runs in MuJoCo simulation at ~500 Hz with a live 3D viewer and browser dashboard. No external planning library required.

---

## What It Does

A pick-and-place command triggers a multi-stage intelligent pipeline: the gripper self-measures from the robot model, a cross-section profiler finds the optimal grip point on the target object, the scene analyzer computes collision-free transit corridors accounting for the full gripper envelope, and a 9-phase sequencer executes the motion with grasp verification, adaptive height selection, and automatic retry. If the target is buried in a stack, CHIRON decomposes the task and clears obstructing objects first.

---

## Self-Measuring Gripper

CHIRON inspects the MuJoCo model at startup and measures its own hand: finger offset from the hand body, pad contact height, pad bottom extent, maximum and minimum opening, and actuator range. Every dimension is read from the model geometry.

This means swapping the robot model (Franka Panda → SO-ARM 101 → any MJCF arm) requires zero code changes. CHIRON re-measures itself and adapts all grasp computations to the new gripper.

```
Gripper measured: finger_offset=0.0962m, pad_bottom=0.1084m,
                  opening=[0.0000, 0.0800]m, pad_height=0.0331m
```

No hardcoded gripper dimensions exist anywhere in the codebase.

---

## Geometry-Based Grasp Computation

Given an object's shape and dimensions (from BROTEUS in production, from MuJoCo ground truth in simulation), the grasp computer finds the optimal grip through cross-section profiling.

**The algorithm:** Sample the object's width at 20 heights. Find all heights where the width fits the gripper. Select the widest cross-section (maximum contact area, maximum friction). Clamp so the entire finger pad contacts the surface — never grip at the edge where half the pad is in empty air.

This is one generic algorithm with no per-shape branches:

- **Box/Cylinder:** Constant width everywhere → highest valid point wins (least hangover), clamped by pad height
- **Sphere:** Equator wins naturally (widest cross-section), gripper opens past the maximum diameter during approach

**Grip force:** Close fully. The object physically blocks the fingers. The actuator force against the object IS the grip force. This is how real parallel-jaw grippers work.

---

## Scene-Aware Motion Planning

Every lateral move goes through strict **UP → ACROSS → DOWN** decomposition. The arm never moves horizontally below safe height. Each sub-move changes only one axis direction, so joint-space interpolation stays close to a straight Cartesian line.

### Transit Height Intelligence

CHIRON computes three different transit heights depending on context:

- **Approach height:** Clears all objects with the full gripper envelope (hand body + finger pads below). Prevents the fingers from clipping objects during horizontal approach.
- **Carrying height:** Path-specific corridor clearance. Only clears objects near the actual transit path, not everything on the table. Accounts for how far the carried object hangs below the gripper.
- **Adaptive height:** If the ideal carrying height is unreachable at the arm's extent, steps down in 2cm increments until finding a height that's both IK-reachable and physically safe. The floor is the physical minimum: table surface + hangover + gap.

### Retreat Logic

Every retreat goes UP: `max(carrying_z, place_z, approach_z)`. After stacking an object at z=0.61, the arm retreats to z=0.70 (the carrying height), never descends through what it just built.

### Safe Return to Ready

After every operation, the arm returns to a neutral pose through a safe path: lift at current XY → transit horizontally to above the arm base → then joint-space move to ready. Never sweeps through the workspace.

---

## Smart Task Decomposition

When the target object is buried in a stack, CHIRON automatically plans a multi-step sequence.

**Example:** Command "pick green_cylinder" when a sphere sits on top of it.

```
TASK DECOMPOSITION: 'green_cylinder' has 1 object(s) above it: ['blue_sphere'].
  Clearing stack first.
CLEARING: Moving 'blue_sphere' to temp [0.35, 0.20]
  → picks sphere, places at computed clear spot
TARGET: Picking 'green_cylinder' → [desired location]
  → now unobstructed, picks cylinder directly
```

The clear spot is computed by searching a grid within the arm's workspace for the position farthest from all existing objects and the final target location. One user command, multiple intelligent sub-steps.

---

## Failure Recovery

### Pre-Check

Before touching anything, the sequencer solves IK for all 6 waypoints (pre-grasp, grasp, lift, transit, place, retreat). If any is unreachable, it fails immediately with a clear message. No object gets disturbed.

### Grasp Verification

After closing the gripper, a small test lift checks if the object moved with the gripper. A second verification after the full lift catches objects that slip during ascent. If either check fails, the arm releases, retreats safely, and returns to ready.

### Automatic Retry

On grip failure, the system waits for physics to settle, re-scans the scene to find where the object ended up, and retries from the new position. Up to 3 attempts. Unreachable-position failures don't trigger retries since they'd fail again.

### Safe Recovery

If anything fails mid-sequence while holding an object: open gripper → retreat upward (never down) → return to ready via safe path. The arm is always left in a clean state for the next command.

---

## Trajectory Recording

Every pick-and-place sequence records joint positions, velocities, efforts, end-effector pose, gripper state, and the active sequencer phase at every timestep. This data feeds directly into DAEDALUS for physics discovery and sim-to-real calibration.

```json
{
  "frame_count": 318,
  "duration": 21.4,
  "phases_seen": ["open_gripper", "move_to_pregrasp", "descend_to_grasp",
                  "close_gripper", "lift", "transit", "descend_to_place",
                  "open_gripper_release", "retreat"]
}
```

---

## Live Dashboard

A browser-based control panel at `localhost:8200` provides:

- Real-time joint states and end-effector pose
- Object selector and custom Place X/Y sliders
- Moveable green target zone (updates in the 3D viewer in real-time)
- Sequencer status with phase progress bar
- Reset Scene button (cancels active tasks, resets physics, cleans state)
- Event log

---

## Architecture

```
┌──────────────────────────────────────────────────────┐
│                   CHIRON Server                      │
│                 FastAPI · Port 8200                  │
│                                                      │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────┐   │
│  │  Gripper    │  │    Grasp     │  │   Scene    │   │
│  │  Model      │  │  Computer    │  │ Awareness  │   │
│  │ (measured)  │  │ (profiling)  │  │ (spatial)  │   │
│  └──────┬──────┘  └──────┬───────┘  └─────┬──────┘   │
│         │                │                │          │
│  ┌──────▼────────────────▼────────────────▼──────┐   │
│  │              Pick-and-Place Sequencer         │   │
│  │   Pre-check · 9-phase · Verify · Retry · Safe │   │
│  └─────────────────────┬─────────────────────────┘   │
│  ┌─────────────────────▼─────────────────────────┐   │
│  │            Damped Least-Squares IK            │   │
│  │          1000 iter · 5mm tolerance            │   │
│  └─────────────────────┬─────────────────────────┘   │
│  ┌─────────────────────▼─────────────────────────┐   │
│  │              RobotBackend (ABC)               │   │
│  │     MuJoCoBackend │ ServoBackend (planned)    │   │
│  └─────────────────────┬─────────────────────────┘   │
│  ┌─────────────────────▼─────────────────────────┐   │
│  │           Trajectory Recorder                 │   │
│  │         → DAEDALUS physics discovery          │   │
│  └───────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────┘
                         │
                  WebSocket / REST
                         │
┌────────────────────────▼─────────────────────────────┐
│              Live Dashboard + 3D Viewer              │
│           Browser · localhost:8200 + MuJoCo          │
└──────────────────────────────────────────────────────┘
```

---

## Part of ORION

CHIRON is one piece of a larger system:

```
ORION (Brain · Port 8000)
 ├── ATHENA    — Navigation · Procedural terrain · A* pathfinding
 ├── BROTEUS   — Perception · Grasp intelligence · Gestures & animations
 ├── CHIRON    — Motor cortex · ROS 2 bridge · Hardware abstraction
 ├── DAEDALUS  — Self-calibrating physics discovery (SINDy)
 └── RL Pipeline — PPO/SAC in sim · ONNX deployment at 50-200 Hz
```

**BROTEUS sees. ORION decides. CHIRON moves. DAEDALUS calibrates.**

The target hardware is an [SO-ARM 101](https://github.com/TheRobotStudio/SO-ARM100), a 6-DOF arm where BROTEUS provides perception, CHIRON drives the joints, and DAEDALUS closes the sim-to-real gap. CHIRON's `RobotBackend` ABC makes the swap from Franka Panda simulation to real SO-ARM 101 hardware a single config change.

---

## Setup

```bash
# Clone
git clone https://github.com/shtet100/CHIRON.git
cd CHIRON

# Environment
conda create -n CHIRON python=3.11 -y
conda activate CHIRON
pip install mujoco fastapi uvicorn websockets numpy pyyaml

# Run (with 3D viewer)
python main.py --render

# Run (headless, dashboard only)
python main.py
```

Open `http://localhost:8200` for the dashboard. The MuJoCo 3D viewer launches alongside when using `--render`.

MuJoCo Menagerie models are included. No additional downloads required.

---

## Tech Stack

| | |
|:---|:---|
| **Simulation** | MuJoCo 3.x (~500 Hz physics) |
| **Robot Model** | Franka Emika Panda (MuJoCo Menagerie) |
| **IK Solver** | Damped least-squares Jacobian (1000 iter, 5mm tol) |
| **Trajectories** | Quintic smoothstep interpolation |
| **Grasp Planning** | Cross-section profiling, pad-aware clamping |
| **Scene Reasoning** | Path-specific corridor clearance, adaptive height |
| **Server** | FastAPI + WebSocket, Python 3.11 |
| **Frontend** | Single-file vanilla JS/CSS dashboard |
| **Recording** | Per-frame joint state logging for DAEDALUS |

---

## Project Structure

```
CHIRON/
├── main.py                          # Entry point (--render for 3D viewer)
├── config/
│   └── chiron_config.yaml           # Backend, port, scene configuration
├── src/chiron/
│   ├── server.py                    # FastAPI server, REST + WebSocket
│   ├── dashboard.html               # Browser control panel
│   ├── sequencer.py                 # 9-phase pick-and-place + retry + decomposition
│   ├── grasp_computer.py            # Cross-section profiling, optimal grip selection
│   ├── gripper_model.py             # Self-measuring gripper geometry
│   ├── scene_awareness.py           # Spatial reasoning, stacking, corridor clearance
│   ├── ik_solver.py                 # Damped least-squares inverse kinematics
│   ├── trajectory_recorder.py       # Per-frame logging for DAEDALUS
│   └── backends/
│       ├── base.py                  # RobotBackend ABC, RobotState dataclass
│       └── mujoco_backend.py        # MuJoCo simulation backend
├── models/
│   └── mujoco_menagerie/
│       └── franka_emika_panda/
│           ├── panda.xml            # Robot model
│           └── workspace.xml        # Table scene with objects + target zone
└── tests/
    └── test_phase_c0.py
```

---

## Design Decisions

**No hardcoded anything.** Every gripper dimension is measured from the model. Every grasp parameter is computed from object geometry. Every transit height is derived from the scene state and the gripper's physical envelope. Swap the robot model and everything re-computes.

**Geometry beats heuristics.** Cross-section profiling finds the optimal grip point for any convex shape. Pad-aware clamping ensures full finger contact. Corridor-specific clearance avoids over-conservative global heights. These are geometric computations, not tuned constants.

**Fail safely, retry intelligently.** Pre-check all waypoints before starting. Verify the grasp after lifting. Recover cleanly if anything fails mid-sequence. Re-scan the scene and retry from the object's actual position. Never leave the arm in a broken state.

**Swappable backend.** The `RobotBackend` ABC abstracts the physical robot. Today it runs `MuJoCoBackend` in simulation. The planned `ServoBackend` for the SO-ARM 101 implements the same interface. The sequencer, grasp computer, and scene awareness don't change.

**Task-level intelligence.** One user command can decompose into multiple sub-tasks. CHIRON understands that picking an object from the middle of a stack requires clearing what's above it first, computing temporary placement spots, and executing the full sequence autonomously.

---

*Built by Swan Yi Htet & David Young.*
