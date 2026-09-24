"""Camera verification: prove a registered camera is actually usable.

Operator-triggered and bounded. Not monitoring -- see M1.6 for that.
"""

from sentinel_system.verification.enums import ConnectionStatus
from sentinel_system.verification.models import CameraVerification
from sentinel_system.verification.schemas import VerificationResult
from sentinel_system.verification.service import CameraVerificationService

__all__ = [
    "CameraVerification",
    "CameraVerificationService",
    "ConnectionStatus",
    "VerificationResult",
]
