from typing import Dict, Any, Optional
import yaml
import os
from pathlib import Path
import logging
from dataclasses import dataclass

@dataclass
class ModelConfig:
    model_name: str
    api_key: Optional[str] = None
    max_tokens: int = 4096
    temperature: float = 0.7

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
        self.logger = logging.getLogger(__name__)
        self.config_path = config_path or os.getenv('APP_CONFIG', 'config/default_config.yaml')
        self.config = self._load_config()

    def _load_config(self) -> AppConfig:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                config_dict = yaml.safe_load(f)

            model_config = ModelConfig(**config_dict['model'])
            embedding_config = EmbeddingConfig(**config_dict['embedding'])
            database_config = DatabaseConfig(**config_dict['database'])

            return AppConfig(
                model=model_config,
                embedding=embedding_config,
                database=database_config,
                **config_dict.get('app', {})
            )

        except Exception as e:
            self.logger.error(f"Error loading config: {str(e)}")
            raise

    def get_config(self) -> AppConfig:
        """Get the current configuration."""
        return self.config

    def update_config(self, config_dict: Dict[str, Any]) -> None:
        """Update configuration values."""
        try:
            with open(self.config_path, 'w') as f:
                yaml.safe_dump(config_dict, f)
            self.config = self._load_config()
        except Exception as e:
            self.logger.error(f"Error updating config: {str(e)}")
            raise