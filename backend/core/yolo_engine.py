# ---
# CampusAI Suite YOLOv8 Engine
# Owner: Member 2 (Computer vision engineer) & Member 1 (ML Lead)
# ---

import os
import logging
import numpy as np
from typing import List, Dict, Any
from pathlib import Path
from backend.core.config import settings

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("yolo_engine")

_yolo_model = None
_using_fallback = False

def init_yolo_engine() -> None:
    """
    Initializes the YOLOv8 model for prohibited object detection.
    Attempts to load a fine-tuned model 'ml/best.pt' first, falling back to standard 'yolov8n.pt'.
    If ultralytics is missing or model fails to load, gracefully routes to the Synthetic Fallback Engine.
    """
    global _yolo_model, _using_fallback
    
    # Monkeypatch torch.load for PyTorch 2.6 weights_only=False compatibility with Ultralytics DetectionModel
    import torch
    original_torch_load = torch.load
    
    def patched_torch_load(*args, **kwargs):
        kwargs['weights_only'] = False
        return original_torch_load(*args, **kwargs)
        
    try:
        # Apply patch
        torch.load = patched_torch_load
        
        from ultralytics import YOLO
        logger.info("Initializing YOLOv8 engine with PyTorch 2.6 compatibility patch...")
        
        # Look for fine-tuned best.pt, then yolov8n.pt
        model_path = settings.BASE_DIR / "ml" / "best.pt"
        if not model_path.exists():
            model_path = settings.BASE_DIR / "yolov8n.pt"
            logger.info(f"Fine-tuned weights not found. Using standard model: {model_path}")
        else:
            logger.info(f"Using fine-tuned weights: {model_path}")
            
        model = YOLO(str(model_path))
        _yolo_model = model
        _using_fallback = False
        logger.info("YOLOv8 engine successfully loaded.")
    except Exception as err:
        if not settings.USE_AI_FALLBACK:
            logger.error(f"Failed to load YOLOv8: {err}. Fallback disabled, raising exception.")
            raise err
        else:
            logger.warning(f"Could not initialize YOLOv8 ({err}). Falling back to Synthetic YOLO Engine.")
            _using_fallback = True
    finally:
        # Restore original torch.load
        torch.load = original_torch_load

# Initialize the YOLO engine
try:
    init_yolo_engine()
except Exception as e:
    logger.error(f"Initial YOLO engine load error: {e}. Fallback will be used if permitted.")
    _using_fallback = True

def is_using_fallback() -> bool:
    """
    Returns True if the YOLO engine is running in fallback mode.
    """
    return _using_fallback

def detect_prohibited_objects(
    frame: np.ndarray,
    mock_trigger: str = ""
) -> List[Dict[str, Any]]:
    """
    Detects prohibited items (phone, cheat_sheet, book, earphone) in the frame.
    
    Args:
        frame: np.ndarray, current video frame in BGR format.
        mock_trigger: str, manual trigger for simulating anomalies in fallback mode.
                      Values can be: "phone", "cheat_sheet", "book", "earphone" or empty.
                      
    Returns:
        List of dicts representing detected objects:
        [
            {
                "label": "phone" | "cheat_sheet" | "book" | "earphone",
                "confidence": 0.8423,
                "bbox": [x1, y1, x2, y2],
                "score_delta": 50
            },
            ...
        ]
    """
    global _yolo_model, _using_fallback
    
    # Prohibited object scores mapping
    score_mapping = {
        "phone": settings.SCORE_PHONE_DETECTED,
        "cheat_sheet": settings.SCORE_CHEAT_SHEET_DETECTED,
        "book": settings.SCORE_BOOK_DETECTED,
        "earphone": 25  # Earphone standard score delta
    }
    
    if not _using_fallback and _yolo_model is not None:
        try:
            # Run YOLOv8 inference
            # Classes mapping depends on training, we map YOLO detections to standard labels
            results = _yolo_model(frame, verbose=False)
            detections = []
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    conf = float(box.conf[0])
                    if conf < settings.PROCTOR_PHONE_CONFIDENCE:
                        continue
                        
                    cls_id = int(box.cls[0])
                    # Try to get class name from model names
                    cls_name = result.names[cls_id].lower()
                    
                    # Map class name to our standard categories
                    label = None
                    if "phone" in cls_name or "cell" in cls_name:
                        label = "phone"
                    elif "cheat" in cls_name or "paper" in cls_name or "notes" in cls_name:
                        label = "cheat_sheet"
                    elif "book" in cls_name or "textbook" in cls_name or "document" in cls_name:
                        label = "book"
                    elif "ear" in cls_name or "headphone" in cls_name:
                        label = "earphone"
                        
                    if label:
                        bbox = box.xyxy[0].cpu().numpy().astype(int).tolist()
                        detections.append({
                            "label": label,
                            "confidence": round(conf, 4),
                            "bbox": bbox,
                            "score_delta": score_mapping[label]
                        })
            return detections
        except Exception as err:
            logger.warning(f"YOLOv8 inference failed: {err}. Routing to Fallback.")
            
    # --- SYNTHETIC FALLBACK YOLO ENGINE ---
    # In fallback mode, we can simulate detections based on a mock trigger
    # (e.g. passed from the Streamlit web dashboard to simulate a student pulling out a phone).
    detections = []
    
    if mock_trigger in score_mapping:
        h, w = frame.shape[:2]
        # Simulate a bounding box in the bottom right where a student holds the item
        bbox = [int(w * 0.6), int(h * 0.5), int(w * 0.9), int(h * 0.9)]
        detections.append({
            "label": mock_trigger,
            "confidence": 0.8872,
            "bbox": bbox,
            "score_delta": score_mapping[mock_trigger]
        })
        
    return detections
