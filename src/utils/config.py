from typing import Dict, Any, Optional
import yaml
import os
from pathlib import Path
import logging
from dataclasses import dataclass
from dotenv import load_dotenv


@dataclass
class ModelConfig:
    model_name: str
    type: str = "azure_openai"
    api_key: Optional[str] = None
    api_version: Optional[str] = None
    max_tokens: int = 4096
    temperature: float = 0.7
    api_base: Optional[str] = None
    deployment_name: Optional[str] = None
    audience: Optional[str] = None
    organization: Optional[str] = None


@dataclass
class EmbeddingConfig:
    model_name: str
    dimension: int
    batch_size: int = 32


@dataclass
class DatabaseConfig:
    db_url: str
    pool_size: int = 10
    max_queries: int = 50000


@dataclass
class AppConfig:
    model: ModelConfig
    embedding: EmbeddingConfig
    database: DatabaseConfig
    chunk_size: int = 1000
    chunk_overlap: int = 200
    vector_store_path: str = "data/vector_store"
    max_context_length: int = 4000
    supported_file_types: list = None


class ConfigManager:
    def __init__(self, config_path: str = None):
        """Initialize ConfigManager with optional config path."""
        self.logger = logging.getLogger(__name__)
        self.config_path = self._get_config_path(config_path)
        self._load_environment()
        self.config = self._load_config()

    def _get_config_path(self, config_path: Optional[str]) -> str:
        """Get configuration file path with fallbacks."""
        if config_path:
            return config_path
        return os.getenv('APP_CONFIG', 'config/default_config.yaml')

    def _load_environment(self) -> None:
        """Load environment variables from .env file if it exists."""
        env_path = Path('.env')
        if env_path.exists():
            load_dotenv(env_path)
            self.logger.info(f"Loaded environment variables from {env_path}")
        else:
            self.logger.warning("No .env file found in current directory")

    def _resolve_env_vars(self, value: Any) -> Any:
        """Recursively resolve environment variables in configuration values."""
        if isinstance(value, str) and value.startswith('$'):
            # Remove $ prefix and any curly braces
            env_var = value.lstrip('$').strip('{}')
            if env_value := os.getenv(env_var):
                return env_value
            self.logger.warning(f"Environment variable {env_var} not found")
            return value

        if isinstance(value, dict):
            return {k: self._resolve_env_vars(v) for k, v in value.items()}

        if isinstance(value, list):
            return [self._resolve_env_vars(item) for item in value]

        return value

    def _load_config(self) -> AppConfig:
        """Load and parse configuration from YAML file."""
        try:
            config_dict = self._read_yaml_file()
            resolved_config = self._resolve_env_vars(config_dict)
            return self._create_config_objects(resolved_config)
        except Exception as e:
            self.logger.error(f"Error loading config: {str(e)}")
            raise

    def _read_yaml_file(self) -> dict:
        """Read and parse YAML configuration file."""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            self.logger.error(f"Error reading config file {self.config_path}: {str(e)}")
            raise

    def _create_config_objects(self, config_dict: dict) -> AppConfig:
        """Create configuration objects from dictionary."""
        return AppConfig(
            model=ModelConfig(**config_dict['model']),
            embedding=EmbeddingConfig(**config_dict['embedding']),
            database=DatabaseConfig(**config_dict['database']),
            **config_dict.get('app', {})
        )

    def get_config(self) -> AppConfig:
        """Get the current configuration."""
        return self.config

    def update_config(self, config_dict: Dict[str, Any]) -> None:
        """Update configuration file and reload configuration."""
        try:
            resolved_config = self._resolve_env_vars(config_dict)
            self._write_yaml_file(resolved_config)
            self.config = self._load_config()
        except Exception as e:
            self.logger.error(f"Error updating config: {str(e)}")
            raise

    def _write_yaml_file(self, config_dict: dict) -> None:
        """Write configuration dictionary to YAML file."""
        try:
            with open(self.config_path, 'w') as f:
                yaml.safe_dump(config_dict, f, default_flow_style=False)
        except Exception as e:
            self.logger.error(f"Error writing config file {self.config_path}: {str(e)}")
            raise