from abc import abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import pickle
import shutil
import threading
from typing import Any, Iterator, Protocol, Self, runtime_checkable

@dataclass(slots=True)
class ModelMeta:
    """
    Model metadata (logically immutable, structurally normalized).
    """

    name: str
    created_at: datetime = field(default=datetime.now(timezone.utc)) 

    ident_names: list[str] | None = None
    label_name: str | None = None

    feature_names: list[str] = field(default_factory=list)
    importance: dict[str, float] = field(default_factory=dict)

    metric_name: str | None = None
    metric_value: float | None = None

    tags: list[str] = field(default_factory=list)
    version: str = "0.0.1"
    framework: str | None = None
    description: str = ""

    # ---------------- normalization ----------------

    def __post_init__(self) -> None:
        """
        Enforce structural invariants once.
        """
        if self.ident_names is not None:
            self.ident_names = list(self.ident_names)

        self.feature_names = list(self.feature_names)
        self.tags = list(self.tags)
        self.importance = dict(self.importance)

    # ---------------- interface ----------------

    def to_dict(self) -> dict[str, Any]:
        """JSON / storage safe representation."""
        return {
            "name": self.name,
            "created_at": self.created_at.isoformat(),
            "ident_names": self.ident_names,
            "label_name": self.label_name,
            "feature_names": self.feature_names,
            "importance": self.importance,
            "metric_name": self.metric_name,
            "metric_value": self.metric_value,
            "tags": self.tags,
            "version": self.version,
            "framework": self.framework,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        return cls(
            name=data["name"],
            created_at=datetime.fromisoformat(data["created_at"]),
            ident_names=data.get("ident_names"),
            label_name=data.get("label_name"),
            feature_names=data.get("feature_names", []),
            importance=data.get("importance", {}),
            metric_name=data.get("metric_name"),
            metric_value=data.get("metric_value"),
            tags=data.get("tags", []),
            version=data.get("version", "0.0.1"),
            framework=data.get("framework"),
            description=data.get("description", ""),
        )

@dataclass(slots=True)
class ModelCard[T]:
    """
    Generic container for model + metadata.
    
    Type Parameters:
        T: The type of the model being stored
    
    Attributes:
        model: The actual model object
        meta: Associated metadata
    """
    model: T
    meta: ModelMeta

@runtime_checkable
class Storage[T](Protocol):
    """
    Storage protocol interface.
    
    All storage backends must implement these methods.
    """
    
    @abstractmethod
    def save(self, name: str, container: ModelCard[T], *, overwrite: bool = False) -> None:
        """
        Save a model container.
        
        Args:
            name: Unique identifier for the model
            container: The model container to store
            overwrite: If True, overwrite existing model with same name
            
        Raises:
            ModelExistsError: If model exists and overwrite=False
            StorageBackendError: If storage operation fails
        """
        ...
    
    @abstractmethod
    def load(self, name: str) -> T:
        """
        Load a model container.
        
        Args:
            name: Unique identifier for the model
            
        Returns:
            The stored model container
            
        Raises:
            ModelNotFoundError: If model doesn't exist
            StorageBackendError: If load operation fails
        """
        ...
    
    @abstractmethod
    def exists(self, name: str) -> bool:
        """Check if a model exists."""
        ...
    
    @abstractmethod
    def delete(self, name: str) -> None:
        """
        Delete a model.
        
        Raises:
            ModelNotFoundError: If model doesn't exist
        """
        ...
    
    @abstractmethod
    def list(self) -> list[str]:
        """List all stored model names."""
        ...
    
    @abstractmethod
    def get_meta(self, name: str) -> ModelMeta:
        """
        Get metadata without loading the full model.
        
        Args:
            name: Model identifier
            
        Returns:
            Model metadata
        """
        ...

        
class FSStorage[T]:
    """
    File-based storage implementation using pickle.
    
    Stores models as .pkl files and metadata as .json files for
    efficient metadata retrieval without full model loading.
    
    Thread-safe for concurrent operations.
    """

    def __init__(self, base_dir: Path | str, *, separate_meta: bool = True):
        """
        Initialize file system storage.
        
        Args:
            base_dir: Directory to store models
            separate_meta: If True, store metadata separately for fast access
        """
        self.base_dir = Path(base_dir)
        self.separate_meta = separate_meta
        self._lock = threading.RLock()
        
        # Create directory if it doesn't exist
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        if self.separate_meta:
            self._meta_dir = self.base_dir / '.meta'
            self._meta_dir.mkdir(exist_ok=True)
    
    def _model_path(self, name: str) -> Path:
        """Get path to model file."""
        return self.base_dir / f"{name}.pkl"
    
    def _meta_path(self, name: str) -> Path:
        """Get path to metadata file."""
        if self.separate_meta:
            return self._meta_dir / f"{name}.json"
        return self._model_path(name)
    
    def save(self, name: str, container: ModelCard[T], *, overwrite: bool = False) -> None:
        """Save model to filesystem."""
        with self._lock:
            model_path = self._model_path(name)
            
            if not overwrite and model_path.exists():
                raise MemoryError(f"Model '{name}' already exists. Use overwrite=True to replace.")
            
            try:
                # Save model
                with open(model_path, "wb") as f:
                    pickle.dump(container, f)
                
                # Save metadata separately if enabled
                if self.separate_meta:
                    meta_path = self._meta_path(name)
                    with open(meta_path, "w") as f:
                        json.dump(container.meta.to_dict(), f, indent=2)
                        
            except (IOError, OSError) as e:
                raise SystemError(f"Failed to save model '{name}': {e}")
    
    def load(self, name: str) -> ModelCard[T]:
        """Load model from filesystem."""
        model_path = self._model_path(name)
        
        if not model_path.exists():
            raise MemoryError(f"Model '{name}' not found")
        
        try:
            with open(model_path, "rb") as f:
                return pickle.load(f)
        except (IOError, OSError, pickle.PickleError) as e:
            raise SystemError(f"Failed to load model '{name}': {e}")
    
    def exists(self, name: str) -> bool:
        """Check if model exists."""
        return self._model_path(name).exists()
    
    def delete(self, name: str) -> None:
        """Delete model from filesystem."""
        with self._lock:
            model_path = self._model_path(name)
            
            if not model_path.exists():
                raise MemoryError(f"Model '{name}' not found")
            
            try:
                model_path.unlink()
                
                # Also delete metadata if separate
                if self.separate_meta:
                    meta_path = self._meta_path(name)
                    if meta_path.exists():
                        meta_path.unlink()
                        
            except (IOError, OSError) as e:
                raise SystemError(f"Failed to delete model '{name}': {e}")
    
    def list(self) -> list[str]:
        """List all stored model names."""
        models = []
        for path in self.base_dir.glob("*.pkl"):
            models.append(path.stem)
        return sorted(models)
    
    def get_meta(self, name: str) -> ModelMeta:
        """Get metadata without loading full model."""
        if self.separate_meta:
            meta_path = self._meta_path(name)
            if not meta_path.exists():
                raise MemoryError(f"Model '{name}' not found")
            
            try:
                with open(meta_path, "r") as f:
                    return ModelMeta.from_dict(json.load(f))
            except (IOError, OSError, json.JSONDecodeError) as e:
                raise SystemError(f"Failed to load metadata for '{name}': {e}")
        else:
            # Fall back to loading full model
            return self.load(name).meta
    
    def clear(self) -> None:
        """Remove all stored models."""
        with self._lock:
            if self.base_dir.exists():
                shutil.rmtree(self.base_dir)
                self.base_dir.mkdir(parents=True, exist_ok=True)
                if self.separate_meta:
                    self._meta_dir = self.base_dir / '.meta'
                    self._meta_dir.mkdir(exist_ok=True)
    
    def get_size(self, name: str) -> int:
        """Get size of stored model in bytes."""
        model_path = self._model_path(name)
        if not model_path.exists():
            raise MemoryError(f"Model '{name}' not found")
        return model_path.stat().st_size


class ModelStore:
    """
    High-level model storage manager.
    
    Provides a unified interface for model lifecycle management
    with support for tagging, searching, and versioning.
    """
    
    def __init__(self, storage: Storage[Any]):
        """
        Initialize model store.
        
        Args:
            storage: Storage backend (defaults to InMemoryStorage)
        """
        self._storage = storage
    
    def register[T](
        self,
        card: ModelCard[T],
        *,
        overwrite:bool=False,
    ):
        """
        Register a new model with metadata.
        
        Args:
            model: The model object to store
            name: Unique identifier
            feature_names: List of feature column names
            metric_name: Evaluation metric name
            metric_value: Evaluation metric value
            tags: Categorization tags
            framework: ML framework used
            description: Human-readable description
            version: Model version
            overwrite: Whether to overwrite existing
            
        Returns:
            The created model container
        """
        self._storage.save(card.meta.name, card, overwrite=overwrite)
    
    def retrieve(self, name: str) -> ModelCard[Any]:
        """Retrieve a model by name."""
        return self._storage.load(name)
    
    def remove(self, name: str) -> None:
        """Remove a model."""
        self._storage.delete(name)
    
    def is_registered(self, name: str) -> bool:
        """Check if model is registered."""
        return self._storage.exists(name)
    
    def list_models(self) -> list[str]:
        """List all registered model names."""
        return self._storage.list()
    
    def get_metadata(self, name: str) -> ModelMeta:
        """Get model metadata without loading the model."""
        return self._storage.get_meta(name)
    
    def find_by_tag(self, tag: str) -> list[str]:
        """Find all models with a specific tag."""
        matching = []
        for name in self._storage.list():
            meta = self._storage.get_meta(name)
            if tag in meta.tags:
                matching.append(name)
        return matching
    
    def find_by_framework(self, framework: str) -> list[str]:
        """Find all models by framework."""
        matching = []
        for name in self._storage.list():
            meta = self._storage.get_meta(name)
            if meta.framework == framework:
                matching.append(name)
        return matching
    
    def export_metadata(self, name: str, path: Path | str) -> None:
        """Export model metadata to JSON file."""
        meta = self._storage.get_meta(name)
        path = Path(path)
        with open(path, 'w') as f:
            json.dump(meta.to_dict(), f, indent=2)
    
    def __iter__(self) -> Iterator[str]:
        """Iterate over model names."""
        return iter(self._storage.list())
    
    def __contains__(self, name: str) -> bool:
        """Check if model exists using 'in' operator."""
        return self._storage.exists(name)
    
    def __len__(self) -> int:
        """Return number of stored models."""
        return len(self._storage.list())