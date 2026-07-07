# ---
# CampusAI Suite — InsightFace Diagnostic Script
# Tests whether InsightFace actually loads and detects faces in student photos.
#
# Run from project root with virtual env active:
#   python ml/diagnose_insightface.py
# ---

import sys
import os
import cv2
import numpy as np
from pathlib import Path

# Add workspace to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

def main():
    print("=" * 65)
    print("  InsightFace / ArcFace Diagnostic Tool")
    print("=" * 65)
    print()

    # --- Step 1: Check if insightface is installed ---
    print("Step 1: Checking insightface installation...")
    try:
        import insightface
        print(f"  ✅ insightface version: {insightface.__version__}")
    except ImportError as e:
        print(f"  ❌ insightface is NOT installed: {e}")
        print(f"  FIX: pip install insightface")
        return

    # --- Step 2: Check ONNX runtime ---
    print()
    print("Step 2: Checking ONNX Runtime...")
    try:
        import onnxruntime as ort
        print(f"  ✅ onnxruntime version: {ort.__version__}")
        providers = ort.get_available_providers()
        print(f"  Available providers: {providers}")
        if 'CUDAExecutionProvider' in providers:
            print(f"  ✅ CUDA GPU acceleration available")
        else:
            print(f"  ⚠️ No CUDA — will run on CPU (slower but functional)")
    except ImportError as e:
        print(f"  ❌ onnxruntime is NOT installed: {e}")
        print(f"  FIX: pip install onnxruntime   (or onnxruntime-gpu for CUDA)")
        return

    # --- Step 3: Try loading the buffalo_l model ---
    print()
    print("Step 3: Loading InsightFace buffalo_l model...")
    try:
        from insightface.app import FaceAnalysis
        app = FaceAnalysis(name='buffalo_l')
        try:
            app.prepare(ctx_id=0, det_size=(640, 640))
            print(f"  ✅ Model loaded on CUDA GPU")
        except Exception as gpu_err:
            print(f"  ⚠️ GPU failed ({gpu_err}), trying CPU...")
            app.prepare(ctx_id=-1, det_size=(640, 640))
            print(f"  ✅ Model loaded on CPU")
    except Exception as e:
        print(f"  ❌ FAILED to load buffalo_l: {e}")
        print()
        print("  This is likely because the model files haven't been downloaded.")
        print("  InsightFace downloads them automatically on first run, but it")
        print("  needs internet access and the correct cache directory.")
        print()
        print(f"  Default model cache: ~/.insightface/models/buffalo_l/")
        home_models = Path.home() / ".insightface" / "models" / "buffalo_l"
        if home_models.exists():
            files = list(home_models.iterdir())
            print(f"  Cache directory exists with {len(files)} files:")
            for f in files:
                print(f"    {f.name} ({f.stat().st_size / 1024:.0f} KB)")
        else:
            print(f"  Cache directory DOES NOT EXIST: {home_models}")
            print(f"  The model needs to be downloaded.")
        return

    # --- Step 4: Test on a known good image (synthetic face) ---
    print()
    print("Step 4: Testing detection on a synthetic test image...")
    # Create a simple image that is unlikely to have a face
    test_img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(test_img, (200, 100), (440, 380), (180, 180, 180), -1)
    faces = app.get(test_img)
    print(f"  Synthetic image (no face): detected {len(faces)} faces (expected: 0)")

    # --- Step 5: Test on actual student photos ---
    print()
    print("Step 5: Testing on actual student photos...")
    from backend.core.config import settings
    students_dir = settings.DATA_DIR / "students"
    
    if not students_dir.exists():
        print(f"  ❌ Students directory not found: {students_dir}")
        return
    
    folders = sorted([f for f in students_dir.iterdir() if f.is_dir()])
    if not folders:
        print(f"  ❌ No student folders found in {students_dir}")
        return

    print(f"  Found {len(folders)} student folders.")
    print()
    
    total_photos = 0
    total_faces_found = 0
    students_with_faces = 0
    students_without_faces = 0
    
    for folder in folders:
        parts = folder.name.split("_", 1)
        sid = parts[0] if len(parts) >= 2 else folder.name
        name = parts[1] if len(parts) >= 2 else "unknown"
        
        images = sorted([p for p in folder.iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png')])
        if not images:
            print(f"  {sid} {name}: 0 photos")
            students_without_faces += 1
            continue
        
        faces_in_student = 0
        for img_path in images[:3]:  # Test up to 3 photos per student
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"    ❌ Failed to read: {img_path.name}")
                continue
            
            total_photos += 1
            h, w = img.shape[:2]
            
            faces = app.get(img)
            
            if faces:
                best = max(faces, key=lambda f: float(f.det_score))
                score = float(best.det_score)
                bbox = best.bbox.astype(int).tolist()
                emb_norm = float(np.linalg.norm(best.embedding))
                print(f"  {sid} {name}/{img_path.name}: {w}x{h} → {len(faces)} face(s), best score={score:.4f}, bbox={bbox}, emb_norm={emb_norm:.2f} ✅")
                faces_in_student += 1
                total_faces_found += 1
            else:
                print(f"  {sid} {name}/{img_path.name}: {w}x{h} → 0 faces ❌")
                
                # Try with different det_size to catch small faces
                app_small = FaceAnalysis(name='buffalo_l')
                try:
                    app_small.prepare(ctx_id=-1, det_size=(320, 320))
                    faces_small = app_small.get(img)
                    if faces_small:
                        print(f"    ↳ Retry with det_size=320: found {len(faces_small)} face(s) ✅")
                        total_faces_found += 1
                        faces_in_student += 1
                except Exception:
                    pass
        
        if faces_in_student > 0:
            students_with_faces += 1
        else:
            students_without_faces += 1

    print()
    print("=" * 65)
    print("  DIAGNOSTIC SUMMARY")
    print("=" * 65)
    print(f"  Total photos tested:     {total_photos}")
    print(f"  Faces detected:          {total_faces_found}")
    print(f"  Students with face:      {students_with_faces}")
    print(f"  Students without face:   {students_without_faces}")
    print()
    
    if total_faces_found == 0 and total_photos > 0:
        print("  ⚠️ ZERO faces detected in ANY photo!")
        print("  Possible causes:")
        print("    1. Images are synthetic colored blocks (mock data), not real face photos")
        print("    2. Image resolution too low (< 100x100)")
        print("    3. Faces are too small in the image (< 30x30 pixel face region)")
        print("    4. ONNX model files are corrupted")
        print()
        print("  Check sample image manually:")
        if folders:
            sample_imgs = list(folders[0].iterdir())
            if sample_imgs:
                simg = cv2.imread(str(sample_imgs[0]))
                if simg is not None:
                    h, w = simg.shape[:2]
                    print(f"    File: {sample_imgs[0]}")
                    print(f"    Resolution: {w}x{h}")
                    print(f"    Mean pixel value: {simg.mean():.1f}")
                    is_uniform = np.std(simg) < 30
                    if is_uniform:
                        print(f"    ⚠️ Image appears to be a UNIFORM COLOR BLOCK (std={np.std(simg):.1f})")
                        print(f"    This is synthetic mock data — InsightFace cannot find faces in it.")
                        print(f"    This is EXPECTED behavior for mock data.")
                    else:
                        print(f"    Image has normal pixel variation (std={np.std(simg):.1f})")
    elif students_without_faces > 0:
        print(f"  ⚠️ {students_without_faces} student(s) had no detectable face.")
        print("  Re-take their enrollment photos with:")
        print("    - Good lighting (face clearly visible)")
        print("    - Face occupying at least 30% of the frame")
        print("    - Frontal or slight angle (not profile/back)")
    else:
        print("  ✅ All students have detectable faces. InsightFace is working correctly!")
    
    print()


if __name__ == "__main__":
    main()
