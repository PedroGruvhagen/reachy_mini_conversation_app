"""Face recognition using ONNX Runtime and ArcFace embeddings.

This module provides face detection, embedding extraction, and comparison
for identifying known persons. Uses OpenCV for face detection and
ArcFace ONNX model for face embeddings.
"""

import os
import logging
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from numpy.typing import NDArray

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    ort = None


logger = logging.getLogger(__name__)


# ArcFace model configuration
ARCFACE_MODEL_URL = "https://huggingface.co/garavv/arcface-onnx/resolve/main/arcface.onnx"
ARCFACE_INPUT_SIZE = (112, 112)
EMBEDDING_DIM = 512
SIMILARITY_THRESHOLD = 0.5  # Cosine similarity threshold for match


class FaceRecognitionService:
    """Face recognition service using ArcFace ONNX model.

    Provides face detection, embedding extraction, and similarity comparison
    for identifying known persons.
    """

    def __init__(self, model_cache_dir: Optional[str] = None) -> None:
        """Initialize the face recognition service.

        Args:
            model_cache_dir: Directory to cache the ONNX model. Defaults to ~/.reachy_mini/models.
        """
        if not ONNX_AVAILABLE:
            logger.warning("onnxruntime not available - face recognition disabled")
            self._initialized = False
            return

        if model_cache_dir is None:
            model_cache_dir = str(Path.home() / ".reachy_mini" / "models")

        self.model_cache_dir = Path(model_cache_dir)
        self.model_cache_dir.mkdir(parents=True, exist_ok=True)

        self.model_path = self.model_cache_dir / "arcface.onnx"
        self.session: Optional[ort.InferenceSession] = None
        self._initialized = False

        # OpenCV face detector (Haar cascade - lightweight)
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    def initialize(self) -> bool:
        """Initialize the face recognition model.

        Downloads the ArcFace model if needed and creates the ONNX session.

        Returns:
            True if initialization succeeded.
        """
        if not ONNX_AVAILABLE:
            return False

        try:
            # Download model if not cached
            if not self.model_path.exists():
                logger.info(f"Downloading ArcFace model to {self.model_path}")
                self._download_model()

            # Create ONNX session
            providers = ["CPUExecutionProvider"]
            if "CUDAExecutionProvider" in ort.get_available_providers():
                providers.insert(0, "CUDAExecutionProvider")

            self.session = ort.InferenceSession(
                str(self.model_path),
                providers=providers,
            )

            self._initialized = True
            logger.info(f"Face recognition initialized with providers: {providers}")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize face recognition: {e}")
            return False

    def _download_model(self) -> None:
        """Download the ArcFace ONNX model from Hugging Face."""
        import urllib.request

        logger.info(f"Downloading ArcFace model from {ARCFACE_MODEL_URL}")
        urllib.request.urlretrieve(ARCFACE_MODEL_URL, str(self.model_path))
        logger.info("ArcFace model downloaded successfully")

    def detect_faces(self, frame: NDArray[np.uint8]) -> List[Tuple[int, int, int, int]]:
        """Detect faces in a frame.

        Args:
            frame: BGR image from OpenCV.

        Returns:
            List of face bounding boxes as (x, y, w, h) tuples.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60),
        )
        return [tuple(f) for f in faces]

    def extract_embedding(self, frame: NDArray[np.uint8]) -> Optional[NDArray[np.float32]]:
        """Extract face embedding from a frame containing a face.

        Args:
            frame: BGR image from OpenCV, ideally cropped to face.

        Returns:
            512-dimensional embedding vector, or None if extraction failed.
        """
        if not self._initialized or self.session is None:
            return None

        try:
            # Detect face and crop
            faces = self.detect_faces(frame)
            if not faces:
                return None

            # Use the largest face
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            face_crop = frame[y:y+h, x:x+w]

            # Preprocess for ArcFace
            face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            face_resized = cv2.resize(face_rgb, ARCFACE_INPUT_SIZE)

            # Normalize to [-1, 1]
            face_normalized = (face_resized.astype(np.float32) - 127.5) / 128.0

            # Add batch dimension and transpose to NCHW
            face_input = np.transpose(face_normalized, (2, 0, 1))
            face_input = np.expand_dims(face_input, axis=0)

            # Run inference
            input_name = self.session.get_inputs()[0].name
            outputs = self.session.run(None, {input_name: face_input})

            embedding = outputs[0][0]

            # Normalize embedding
            embedding = embedding / np.linalg.norm(embedding)

            return embedding.astype(np.float32)

        except Exception as e:
            logger.error(f"Failed to extract face embedding: {e}")
            return None

    def compare_embeddings(
        self,
        embedding1: NDArray[np.float32],
        embedding2: NDArray[np.float32],
    ) -> float:
        """Compare two face embeddings using cosine similarity.

        Args:
            embedding1: First face embedding (normalized).
            embedding2: Second face embedding (normalized).

        Returns:
            Cosine similarity score (0.0 to 1.0, higher = more similar).
        """
        # Both embeddings should already be normalized
        similarity = float(np.dot(embedding1, embedding2))
        return max(0.0, min(1.0, similarity))

    def find_matching_person(
        self,
        current_embedding: NDArray[np.float32],
        known_persons: List[Dict[str, Any]],
    ) -> Optional[Tuple[int, str, float]]:
        """Find the best matching person from known persons.

        Args:
            current_embedding: Embedding from current camera frame.
            known_persons: List of dicts with 'id', 'name', 'embedding' keys.

        Returns:
            Tuple of (person_id, name, confidence) or None if no match.
        """
        best_match: Optional[Tuple[int, str, float]] = None
        best_similarity = SIMILARITY_THRESHOLD

        for person in known_persons:
            stored_embedding = person.get("embedding")
            if stored_embedding is None:
                continue

            similarity = self.compare_embeddings(current_embedding, stored_embedding)

            if similarity > best_similarity:
                best_similarity = similarity
                best_match = (person["id"], person["name"], similarity)

        return best_match

    def compute_frame_hash(self, frame: NDArray[np.uint8]) -> str:
        """Compute a hash for the frame (for caching).

        Args:
            frame: BGR image from OpenCV.

        Returns:
            MD5 hash of the frame.
        """
        # Resize to small size for faster hashing
        small = cv2.resize(frame, (64, 64))
        return hashlib.md5(small.tobytes()).hexdigest()

    @property
    def is_initialized(self) -> bool:
        """Check if the service is initialized."""
        return self._initialized


# Global instance (lazy initialization)
_face_recognition_service: Optional[FaceRecognitionService] = None


def get_face_recognition_service() -> FaceRecognitionService:
    """Get or create the global face recognition service instance."""
    global _face_recognition_service
    if _face_recognition_service is None:
        _face_recognition_service = FaceRecognitionService()
    return _face_recognition_service
