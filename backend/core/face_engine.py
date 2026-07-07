# ---
# CampusAI Suite Face Engine (InsightFace buffalo_l & Synthetic Fallback)
# Owner: Member 1 (ML and face recognition lead)
#
# KEY INSIGHT (from diagnostic run 2026-06-01):
#   InsightFace RetinaFace with det_size=(640,640) CANNOT detect faces in
#   close-up portrait photos (2000-4000px) where the face fills >70% of
#   the frame. The face becomes too large relative to the anchor grid.
#
#   det_size=(320,320) DOES detect those faces — proven on all 24 students.
#
#   For CLASSROOM photos (SmartAttend), faces are small/distant and may
#   need det_size=(640,640) for sufficient resolution.
#
#   InsightFace does NOT allow changing det_size after prepare() — it
#   prints "det_size is already set in detection model, ignore".
#
#   Solution: Initialize TWO FaceAnalysis instances at startup:
#     _app_640: det_size=(640,640) for classroom/group photos
#     _app_320: det_size=(320,320) for enrollment portraits / selfies
#   detect_faces() tries _app_640 first, falls back to _app_320 if 0 faces.
# ---

import logging
import numpy as np
import cv2
from typing import List, Tuple, Optional, Dict, Any
from backend.core.config import settings

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("face_engine")

# Global variables for AI models
_app_640 = None       # Primary detector: det_size=(640,640) for classroom photos
_app_320 = None       # Fallback detector: det_size=(320,320) for portraits
_using_fallback = False


def _init_insightface_app(det_size: Tuple[int, int]) -> Any:
    """
    Create and prepare a single InsightFace FaceAnalysis instance.
    Returns the app object, or raises on failure.
    """
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name=settings.FACE_MODEL_NAME)
    try:
        app.prepare(ctx_id=0, det_size=det_size)
        logger.info(f"InsightFace (det_size={det_size}) initialized on GPU.")
    except Exception:
        app.prepare(ctx_id=-1, det_size=det_size)
        logger.info(f"InsightFace (det_size={det_size}) initialized on CPU.")
    return app


def init_face_engine() -> None:
    """
    Initializes InsightFace with TWO detection scales.
    If it fails, falls back to synthetic face engine.
    """
    global _app_640, _app_320, _using_fallback
    
    if not settings.USE_AI_FALLBACK:
        try:
            _app_640 = _init_insightface_app((640, 640))
            _app_320 = _init_insightface_app((320, 320))
            _using_fallback = False
        except Exception as err:
            logger.error(f"Failed to load InsightFace: {err}. AI Fallback is disabled, raising exception.")
            raise err
    else:
        try:
            _app_640 = _init_insightface_app((640, 640))
            _app_320 = _init_insightface_app((320, 320))
            _using_fallback = False
            logger.info("Both InsightFace detectors (640 + 320) initialized successfully.")
        except Exception as err:
            logger.warning(f"Could not initialize InsightFace ({err}). Falling back to Synthetic Face Engine.")
            _using_fallback = True

# Initialize the engine at import time
try:
    init_face_engine()
except Exception as e:
    logger.error(f"Initial face engine load error: {e}. Fallback will be used if permitted.")
    _using_fallback = True

def is_using_fallback() -> bool:
    """
    Returns True if the engine is currently running in synthetic fallback mode.
    """
    return _using_fallback


def _extract_faces_from_app(app, image: np.ndarray) -> List[Dict[str, Any]]:
    """
    Run detection+recognition on an image using a given InsightFace app.
    Returns list of face result dicts.
    """
    faces = app.get(image)
    results = []
    for face in faces:
        bbox = face.bbox.astype(int).tolist()
        embedding = face.embedding.astype(np.float32)
        det_score = float(face.det_score)
        results.append({
            "bbox": bbox,
            "det_score": round(det_score, 4),
            "embedding": embedding,
            "is_synthetic": False
        })
    return results


def detect_faces(image: np.ndarray, allow_synthetic: bool = True) -> List[Dict[str, Any]]:
    """
    Detects faces in the given image using multi-scale detection.
    
    Strategy:
      1. Try with det_size=640 app — best for classroom photos with distant faces.
      2. If 0 faces, retry with det_size=320 app — best for close-up portraits.
      3. If still 0 and allow_synthetic=True, fall back to Haar Cascade / Grid.
    
    Args:
        image: np.ndarray, input image in BGR format.
        allow_synthetic: bool, if False, never use the synthetic fallback.
        
    Returns:
        List of dicts, each containing:
            - 'bbox': [x1, y1, x2, y2] bounding box
            - 'det_score': confidence score of the detection
            - 'embedding': 512-dimensional float32 embedding
            - 'is_synthetic': bool, True if embedding came from fallback
    """
    global _app_640, _app_320, _using_fallback
    
    h, w = image.shape[:2]
    
    if not _using_fallback and _app_640 is not None:
        try:
            # Scale 1: det_size=640 (classroom/group photos with small/distant faces)
            results = _extract_faces_from_app(_app_640, image)
            if results:
                logger.info(f"Detected {len(results)} face(s) at det_size=640 in {w}x{h} image")
                return results
            
            # Scale 2: det_size=320 (portrait/selfie enrollment photos)
            if _app_320 is not None:
                results = _extract_faces_from_app(_app_320, image)
                if results:
                    logger.info(f"Detected {len(results)} face(s) at det_size=320 in {w}x{h} image (portrait retry)")
                    return results
            
            # Both scales failed
            if allow_synthetic and settings.USE_AI_FALLBACK:
                logger.warning(f"InsightFace detected 0 faces in {w}x{h} image at both scales. Routing to Synthetic Fallback.")
            elif not allow_synthetic:
                logger.warning(f"InsightFace detected 0 faces in {w}x{h} image at both scales. allow_synthetic=False, returning empty.")
                return []
            else:
                return []
        except Exception as err:
            logger.warning(f"InsightFace inference failed: {err}.")
            if not allow_synthetic:
                logger.error("InsightFace failed and allow_synthetic=False. Returning empty.")
                return []
            logger.warning("Routing to Fallback.")
    
    # --- SYNTHETIC / HAAR FALLBACK ENGINE ---
    if not allow_synthetic:
        return []
    
    results = []
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_eq = cv2.equalizeHist(gray)
    
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    
    detected_faces = []
    if not face_cascade.empty():
        detected_faces = face_cascade.detectMultiScale(
            gray_eq, scaleFactor=1.05, minNeighbors=3, minSize=(30, 30)
        )
        if len(detected_faces) == 0:
            detected_faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=3, minSize=(20, 20)
            )
    
    if len(detected_faces) > 0:
        for (x, y, w, h) in detected_faces:
            bbox = [int(x), int(y), int(x + w), int(y + h)]
            face_crop = gray[y:y+h, x:x+w]
            embedding = _generate_synthetic_embedding(face_crop)
            results.append({
                "bbox": bbox,
                "det_score": 0.9500,
                "embedding": embedding,
                "is_synthetic": True
            })
    else:
        h_img, w_img = image.shape[:2]
        mock_faces = [
            [int(w_img * 0.25), int(h_img * 0.3), int(w_img * 0.45), int(h_img * 0.6)],
            [int(w_img * 0.55), int(h_img * 0.3), int(w_img * 0.75), int(h_img * 0.6)]
        ]
        for idx, bbox in enumerate(mock_faces):
            x1, y1, x2, y2 = bbox
            face_crop = gray[y1:y2, x1:x2]
            embedding = _generate_synthetic_embedding(face_crop, seed_offset=idx)
            results.append({
                "bbox": bbox,
                "det_score": 0.8800,
                "embedding": embedding,
                "is_synthetic": True
            })
            
    return results

def _generate_synthetic_embedding(face_gray: np.ndarray, seed_offset: int = 0) -> np.ndarray:
    """
    Generates a deterministic 512-dimensional float32 embedding
    from a grayscale face crop. Downsamples to 16x32 and normalizes to unit length.
    """
    try:
        resized = cv2.resize(face_gray, (16, 32), interpolation=cv2.INTER_AREA)
        flat = resized.flatten().astype(np.float32)
        if seed_offset != 0:
            flat = flat + seed_offset * 10.0
        norm = np.linalg.norm(flat)
        if norm > 0:
            flat = flat / norm
        return flat
    except Exception as e:
        logger.error(f"Failed to generate synthetic embedding: {e}")
        vec = np.zeros(512, dtype=np.float32)
        vec[0] = 1.0
        return vec

def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """
    Computes the cosine similarity between two 512-dimensional embeddings.
    """
    if emb1 is None or emb2 is None:
        return 0.0
    
    norm1 = np.linalg.norm(emb1)
    norm2 = np.linalg.norm(emb2)
    
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
        
    dot_product = np.dot(emb1, emb2)
    similarity = dot_product / (norm1 * norm2)
    return float(round(similarity, 4))

def match_face_to_db(
    face_embedding: np.ndarray,
    enrolled_students: List[Any],
    margin: Optional[float] = None,
) -> Tuple[Optional[str], float]:
    """
    Matches a given face embedding against the enrolled student database.

    Returns the best-matching student_id and the top cosine similarity.

    A MARGIN check is applied by default: the top candidate must beat the
    second-best candidate by at least `margin` (settings.FACE_MATCH_MARGIN).
    If the gap is too small the match is rejected and the function returns
    (None, top_similarity) — the caller can still read the raw top score
    for logging, but should treat the face as UNKNOWN.

    This is the core of the "false negative > false positive" competition
    rule: when several enrolled students score close together, we refuse
    to guess rather than pick the least-wrong answer.

    Args:
        face_embedding: 512-d np.ndarray of the live face.
        enrolled_students: list of Student ORM objects exposing
            get_embedding() and student_id.
        margin: optional override for settings.FACE_MATCH_MARGIN. Pass 0.0
            to disable the margin check (e.g. for entry verification where
            a single deliberate photo is being compared).

    Returns:
        (student_id, confidence).  student_id is None when the margin check
        fails or no enrolled students were supplied.
    """
    if margin is None:
        margin = settings.FACE_MATCH_MARGIN

    best_student_id: Optional[str] = None
    best_similarity: float = -1.0
    second_similarity: float = -1.0

    for student in enrolled_students:
        student_emb = student.get_embedding()
        if student_emb is None:
            continue
        sim = cosine_similarity(face_embedding, student_emb)
        if sim > best_similarity:
            second_similarity = best_similarity
            best_similarity = sim
            best_student_id = student.student_id
        elif sim > second_similarity:
            second_similarity = sim

    confidence = max(0.0, best_similarity)

    # Margin rule: refuse to guess when the top two candidates are too close
    if margin > 0.0 and best_student_id is not None:
        gap = best_similarity - max(second_similarity, 0.0)
        if gap < margin:
            logger.debug(
                f"match_face_to_db: rejected {best_student_id} — top2 gap "
                f"{gap:.4f} < margin {margin:.4f} (top={best_similarity:.4f}, "
                f"second={max(second_similarity, 0.0):.4f})"
            )
            return None, confidence

    return best_student_id, confidence


def _upscale_face_crop(face_crop: np.ndarray, target_size: int = None) -> np.ndarray:
    """
    Upscale a small face crop to a target pixel size using cubic interpolation.

    Used as a recovery path for marginal faces (between FACE_MIN_WIDTH and
    FACE_RECOVERY_WIDTH) where ArcFace would otherwise refuse to embed.

    Args:
        face_crop: BGR face crop from the original frame.
        target_size: Target square edge in pixels. Defaults to settings.FACE_UPSCALE_SIZE.

    Returns:
        Upscaled BGR face crop. Returns the original if already large enough.
    """
    if target_size is None:
        target_size = settings.FACE_UPSCALE_SIZE

    if face_crop is None or face_crop.size == 0:
        return face_crop

    h, w = face_crop.shape[:2]
    if max(h, w) >= target_size:
        return face_crop

    return cv2.resize(
        face_crop,
        (target_size, target_size),
        interpolation=cv2.INTER_CUBIC,
    )


def _extract_embedding_from_crop(face_crop: np.ndarray) -> Optional[np.ndarray]:
    """
    Run InsightFace on a (possibly upscaled) BGR face crop to get a 512-d embedding.

    Falls back to the synthetic embedding path if InsightFace is unavailable.
    Returns None if the crop is invalid.
    """
    if face_crop is None or face_crop.size == 0:
        return None

    # Try the primary detector first (handles upscaled crops well),
    # then the 320 detector as a backup.
    for app in (_app_640, _app_320):
        if app is None:
            continue
        try:
            faces = app.get(face_crop)
            if faces and len(faces) > 0:
                emb = getattr(faces[0], "normed_embedding", None)
                if emb is not None:
                    return np.asarray(emb, dtype=np.float32)
        except Exception as e:
            logger.debug(f"InsightFace embedding failed on app: {e}")

    # Fallback path: derive a deterministic synthetic embedding from the crop
    try:
        return _generate_synthetic_embedding(face_crop)
    except Exception as e:
        logger.warning(f"Synthetic embedding fallback also failed: {e}")
        return None


def recognize_with_upscaling(
    face_crop: np.ndarray,
    enrolled_students: List[Any],
    face_width: int = None,
) -> Tuple[Optional[str], float, str]:
    """
    Recognize a face from a BGR crop, applying ROI upscaling for marginal faces.

    Tier logic (driven by settings):
        - face_width < FACE_MIN_WIDTH  : "too_far"   — skip, caller labels "TOO FAR"
        - FACE_MIN_WIDTH  <= face_width < FACE_RECOVERY_WIDTH : "marginal"
            → upscale to FACE_UPSCALE_SIZE, then embed + match
        - face_width >= FACE_RECOVERY_WIDTH : "good"   — direct embed + match

    Args:
        face_crop: BGR face crop (already extracted from the frame).
        enrolled_students: list of Student ORM objects with get_embedding().
        face_width: optional pre-computed face width in pixels; if None it is
            inferred from the crop.

    Returns:
        (student_id, confidence, tier) where tier is "good" | "marginal" | "too_far".
        student_id is None when no confident match is found.
    """
    if face_crop is None or face_crop.size == 0:
        return None, -1.0, "too_far"

    if face_width is None:
        face_width = face_crop.shape[1]

    min_w = settings.FACE_MIN_WIDTH
    rec_w = settings.FACE_RECOVERY_WIDTH

    # Tier 1 — face is too small to attempt recognition at all
    if face_width < min_w:
        return None, -1.0, "too_far"

    # Tier 2 — marginal: upscale before embedding
    if face_width < rec_w:
        tier = "marginal"
        crop_to_embed = _upscale_face_crop(face_crop, target_size=settings.FACE_UPSCALE_SIZE)
    else:
        # Tier 3 — good: embed as-is
        tier = "good"
        crop_to_embed = face_crop

    embedding = _extract_embedding_from_crop(crop_to_embed)
    if embedding is None or not enrolled_students:
        return None, 0.0, tier

    # Per-tier threshold: marginal faces need a HIGHER cosine similarity
    # to be accepted as a real match.  A marginal (upscaled) crop has
    # more interpolation noise, so we trade a few false negatives for
    # far fewer false positives.
    if tier == "good":
        tier_threshold = settings.FACE_MATCH_THRESHOLD_GOOD
    elif tier == "marginal":
        tier_threshold = settings.FACE_MATCH_THRESHOLD_MARGINAL
    else:
        tier_threshold = settings.FACE_MATCH_THRESHOLD_GOOD

    student_id, confidence = match_face_to_db(embedding, enrolled_students)

    # Apply the per-tier confidence floor.  The margin check inside
    # match_face_to_db has already ensured best-vs-second separation.
    if student_id is not None and confidence < tier_threshold:
        logger.debug(
            f"recognize_with_upscaling: rejected tier={tier} match "
            f"student_id={student_id} confidence={confidence:.4f} < "
            f"tier_threshold={tier_threshold:.4f}"
        )
        return None, confidence, tier

    return student_id, confidence, tier

