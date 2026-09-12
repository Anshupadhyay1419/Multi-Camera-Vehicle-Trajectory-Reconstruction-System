"""
Multi-camera layer: the camera registry, video-source abstraction, and the
sequential processing manager.

    from src.cameras import load_camera_registry, create_video_source

Import the manager from src.cameras.manager directly rather than from here
-- it pulls in the whole vision pipeline, which the API process has no
reason to load just to read a camera list.
"""

from src.cameras.models import (
    CameraConfig,
    CameraProgress,
    CameraState,
    SessionState,
    SessionStatus,
    SourceType,
)
from src.cameras.registry import (
    DEFAULT_CAMERA_CONFIG_PATH,
    CameraConfigError,
    CameraRegistry,
    load_camera_registry,
)
from src.cameras.sources import (
    RTSPSource,
    UploadSource,
    VideoSource,
    VideoSourceError,
    create_video_source,
)

__all__ = [
    "CameraConfig",
    "CameraProgress",
    "CameraState",
    "CameraRegistry",
    "CameraConfigError",
    "SessionState",
    "SessionStatus",
    "SourceType",
    "load_camera_registry",
    "DEFAULT_CAMERA_CONFIG_PATH",
    "VideoSource",
    "UploadSource",
    "RTSPSource",
    "VideoSourceError",
    "create_video_source",
]
