from dataclasses import dataclass, field
from typing import List, Optional
import yaml
import logging

@dataclass
class LoggingConfig:
    experiment_name: str = "Default"
    run_name: str = "default_run"
    log_dir: Optional[str] = None
    tensorboard: bool = True
    tags: List[str] = field(default_factory=lambda: ["default"])


@dataclass
class TrainingConfig:
    num_episodes: int = 200
    num_steps: int = 1000
    save_model_interval: int = 50
    target_update_frequency: int = 1000
    batch_size: int = 64
    epsilon_decay: float = 0.995
    epsilon_end: float = 0.01
    epsilon_start: float = 1.0

@dataclass
class EnvironmentConfig:
    camera: int = 0
    render: bool = True
    parameters: List[str] = field(default_factory=lambda: [
        "CAP_PROP_BRIGHTNESS",
        "CAP_PROP_CONTRAST",
        "CAP_PROP_SATURATION",
        "CAP_PROP_EXPOSURE",
    ])


@dataclass
class AgentConfig:
    gamma: float = 0.99
    learning_rate: float = 0.001
    type: str = "QAgent"
    checkpoint: Optional[str] = None
    memory_size: int = 10000


class Config:
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)

    def __init__(self, config_file):
        self.logger = logging.getLogger(self.__class__.__name__)

        # set default values
        self.logging = LoggingConfig()
        self.training = TrainingConfig()
        self.environment = EnvironmentConfig()
        self.agent = AgentConfig()

        if config_file is None:
            return

        # load config from file
        with open(config_file, "r") as f:
            yaml_data = yaml.safe_load(f)

        self._set(self.logging, yaml_data.get("logging", {}))
        self._set(self.training, yaml_data.get("training", {}))
        self._set(self.environment, yaml_data.get("environment", {}))
        self._set(self.agent, yaml_data.get("agent", {}))

    def _set(self, subconfig, data):
        for key, value in data.items():
            if hasattr(subconfig, key):
                setattr(subconfig, key, value)
            else:
                self.logger.warning(f"Unknown key {key} in config")

    def __str__(self):
        return yaml.dump(self.__dict__)
