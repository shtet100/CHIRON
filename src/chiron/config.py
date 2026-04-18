"""
CHIRON configuration loader.

Reads chiron_config.yaml and provides typed access to all settings.
Falls back to sensible defaults if keys are missing.
"""

import yaml
import logging
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

logger = logging.getLogger("chiron.config")


@dataclass
class ServerConfig:
    port: int = 8200
    host: str = "0.0.0.0"
    state_publish_rate_hz: int = 50
    command_timeout_sec: float = 5.0


@dataclass
class MuJoCoConfig:
    model_file: str = "models/mujoco_menagerie/franka_emika_panda/workspace.xml"
    render: bool = False
    dt: float = 0.002


@dataclass
class GenesisConfig:
    scene_file: str = "scenes/so_arm_101.py"
    render: bool = True
    dt: float = 0.002


@dataclass
class LeRobotConfig:
    port: str = "/dev/ttyUSB0"
    baudrate: int = 1000000
    servo_ids: list = field(default_factory=lambda: [1, 2, 3, 4, 5, 6])


@dataclass
class ChironConfig:
    server: ServerConfig = field(default_factory=ServerConfig)
    active_backend: str = "mujoco"
    mujoco: MuJoCoConfig = field(default_factory=MuJoCoConfig)
    genesis: GenesisConfig = field(default_factory=GenesisConfig)
    lerobot: LeRobotConfig = field(default_factory=LeRobotConfig)


def load_config(config_path: str = "config/chiron_config.yaml") -> ChironConfig:
    """Load config from YAML file. Returns defaults if file is missing."""
    path = Path(config_path)
    if not path.exists():
        logger.warning(f"Config file not found at {config_path}, using defaults")
        return ChironConfig()

    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    chiron_raw = raw.get("chiron", {})
    backends_raw = raw.get("backends", {})

    server = ServerConfig(
        port=chiron_raw.get("port", 8200),
        host=chiron_raw.get("host", "0.0.0.0"),
        state_publish_rate_hz=chiron_raw.get("state_publish_rate_hz", 50),
        command_timeout_sec=chiron_raw.get("command_timeout_sec", 5.0),
    )

    mujoco_raw = backends_raw.get("mujoco", {})
    mujoco_cfg = MuJoCoConfig(
        model_file=mujoco_raw.get("model_file", MuJoCoConfig.model_file),
        render=mujoco_raw.get("render", False),
        dt=mujoco_raw.get("dt", 0.002),
    )

    genesis_raw = backends_raw.get("genesis", {})
    genesis_cfg = GenesisConfig(
        scene_file=genesis_raw.get("scene_file", GenesisConfig.scene_file),
        render=genesis_raw.get("render", True),
        dt=genesis_raw.get("dt", 0.002),
    )

    lerobot_raw = backends_raw.get("lerobot", {})
    lerobot_cfg = LeRobotConfig(
        port=lerobot_raw.get("port", "/dev/ttyUSB0"),
        baudrate=lerobot_raw.get("baudrate", 1000000),
        servo_ids=lerobot_raw.get("servo_ids", [1, 2, 3, 4, 5, 6]),
    )

    config = ChironConfig(
        server=server,
        active_backend=backends_raw.get("active", "mujoco"),
        mujoco=mujoco_cfg,
        genesis=genesis_cfg,
        lerobot=lerobot_cfg,
    )

    logger.info(f"Loaded config: backend={config.active_backend}, port={config.server.port}")
    return config
