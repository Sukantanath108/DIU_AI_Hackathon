# ---
# CampusAI Suite ExamShield Streamlit Dashboard
# Owner: Member 4 (Frontend and Integration) & Member 2 (Computer vision engineer)
# Competition Readiness Version — Auto-Attribution + Verified Roster
# ---

import os
import sys
import time
import json
import requests
import cv2
import numpy as np
import streamlit as st
from datetime import datetime
from pathlib import Path

# Add workspace directory to python path to allow importing config
sys.path.append(str(Path(__file__).resolve().parent.parent))
from backend.core.config import settings

# Page config
st.set_page_config(
    page_title="ExamShield AI Proctoring Console",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Minimalist dark/glassmorphic look using CSS)
st.markdown("""
<style>
    .main {
        background-color: #0F172A;
        color: #F8FAFC;
    }
    .stButton>button {
        background: linear-gradient(135deg, #3B82F6 0%, #06B6D4 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 8px 16px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        filter: brightness(1.1);
        transform: translateY(-1px);
    }
    .risk-badge {
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        display: inline-block;
    }
    .risk-green { background-color: rgba(16, 185, 129, 0.15); color: #10B981; border: 1px solid rgba(16, 185, 129, 0.3); }
    .risk-amber { background-color: rgba(245, 158, 11, 0.15); color: #F59E0B; border: 1px solid rgba(245, 158, 11, 0.3); }
    .risk-red { background-color: rgba(239, 68, 68, 0.15); color: #EF4444; border: 1px solid rgba(239, 68, 68, 0.3); animation: blinker 1.5s linear infinite; }
    .risk-gray { background-color: rgba(100, 116, 139, 0.15); color: #64748B; border: 1px solid rgba(100, 116, 139, 0.3); }
    
    @keyframes blinker {
        50% { opacity: 0.5; }
    }
    
    .student-card {
        background: rgba(30, 41, 59, 0.45);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .student-card-gray {
        background: rgba(30, 41, 59, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.04);
        border-radius: 14px;
        padding: 16px;
        margin-bottom: 12px;
        opacity: 0.5;
    }
</style>
""", unsafe_allow_html=True)

# API Configurations
# Internal API URL (container-to-container inside Docker Compose)
API_URL = os.getenv("BACKEND_API_URL", "http://backend:8000")

# External/Browser-facing API URL (for downloads and resource links on the client machine)
BROWSER_API_URL = os.getenv("BROWSER_API_URL", "http://localhost:8000")


# ============================================================
# HELPER: Load student list from mapping file
# ============================================================
def load_student_list():
    """
    Reads data/student_id_mapping.txt and returns a list of 'SXXX - Name' strings.
    Filters out UNKNOWN-* sentinel rows so the user only sees real candidates.
    Falls back to a safe default if the file doesn't exist.
    """
    mapping_file = settings.DATA_DIR / "student_id_mapping.txt"
    students = []
    if mapping_file.exists():
        with open(mapping_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                # Format: "S000 | Name_Name"
                parts = line.split("|", 1)
                if len(parts) == 2:
                    sid = parts[0].strip()
                    name = parts[1].strip()
                    # Skip UNKNOWN-* sentinel rows
                    if sid.upper().startswith("UNKNOWN"):
                        continue
                    students.append(f"{sid} - {name}")
    if not students:
        students = ["S022 - Sukanta_nath"]
    return students


def get_active_student_id():
    """Returns the current active student ID based on mode (used in Demo Mode only)."""
    if st.session_state.get("demo_mode", True):
        sel = st.session_state.get("demo_student_selection", "S022 - Sukanta_nath")
        return sel.split(" - ")[0] if " - " in sel else sel
    return st.session_state.get("selected_student_id", "S001")


# ============================================================
# HELPER: Find nearest recognized student to an object bbox
# ============================================================
def find_nearest_student(object_bbox, recognized_faces):
    """
    Find which VERIFIED recognized student is closest to a detected object.
    Returns student_id or None if no faces are recognized.
    """
    if not recognized_faces:
        return None

    obj_cx = (object_bbox[0] + object_bbox[2]) / 2
    obj_cy = (object_bbox[1] + object_bbox[3]) / 2

    best_student = None
    best_dist = float('inf')

    for student_id, face_info in recognized_faces.items():
        fx1, fy1, fx2, fy2 = face_info["bbox"]
        face_cx = (fx1 + fx2) / 2
        face_cy = (fy1 + fy2) / 2
        dist = ((obj_cx - face_cx)**2 + (obj_cy - face_cy)**2) ** 0.5
        if dist < best_dist:
            best_dist = dist
            best_student = student_id

    return best_student


# ============================================================
# Initializing Session States
# ============================================================
if "exam_session" not in st.session_state:
    st.session_state.exam_session = None
if "webcam_active" not in st.session_state:
    st.session_state.webcam_active = False
if "selected_student_id" not in st.session_state:
    st.session_state.selected_student_id = "S001"
if "gaze_timer_start" not in st.session_state:
    st.session_state.gaze_timer_start = None
if "demo_mode" not in st.session_state:
    st.session_state.demo_mode = True
if "demo_student_selection" not in st.session_state:
    st.session_state.demo_student_selection = "S022 - Sukanta_nath"
if "current_frame_bytes" not in st.session_state:
    st.session_state.current_frame_bytes = None
# Auto-attribution states
if "recognized_faces_cache" not in st.session_state:
    st.session_state.recognized_faces_cache = {}
if "per_student_gaze_timer" not in st.session_state:
    st.session_state.per_student_gaze_timer = {}
if "unknown_face_count" not in st.session_state:
    st.session_state.unknown_face_count = 0
if "verified_student_ids" not in st.session_state:
    st.session_state.verified_student_ids = set()

# Sidebar - Session Initiation
st.sidebar.title("🛡️ ExamShield AI Control")
st.sidebar.markdown("---")

if st.session_state.exam_session is None:
    st.sidebar.subheader("Start Proctoring Session")
    course = st.sidebar.text_input("Course Name / Code", "CSE301")
    hall = st.sidebar.text_input("Exam Hall Room", "Room 402")
    invigilator = st.sidebar.text_input("Invigilator Name", "Dr. Hasan")
    
    if st.sidebar.button("Initialize Proctoring Session"):
        try:
            r = requests.post(f"{API_URL}/exam/session", json={
                "course": course,
                "hall": hall,
                "invigilator": invigilator
            })
            if r.status_code == 200 and r.json()["status"] == "ok":
                st.session_state.exam_session = r.json()["data"]
                st.toast("Proctoring session initialized successfully!", icon="✅")
                st.rerun()
            else:
                st.sidebar.error("Failed to start session. Verify API backend.")
        except Exception as e:
            st.sidebar.error(f"Backend offline: {e}")
else:
    sess = st.session_state.exam_session
    st.sidebar.success(f"⚡ PROCTORING ACTIVE")
    st.sidebar.markdown(f"**Course:** {sess['course']}")
    st.sidebar.markdown(f"**Hall:** {sess['hall']}")
    st.sidebar.markdown(f"**Session ID:** #{sess['exam_session_id']}")
    st.sidebar.markdown("---")

    # ============================================================
    # EXAM ROSTER (Phase 15)
    # ============================================================
    with st.sidebar.expander("📋 Exam Roster (Candidates)", expanded=False):
        st.markdown("Select which students are taking this exam. Only selected students appear in the Risk Grid.")
        
        student_list = load_student_list()
        
        # Initialize roster state
        if "roster_set" not in st.session_state:
            st.session_state.roster_set = False
        
        selected_candidates = st.multiselect(
            "Exam Candidates:",
            options=student_list,
            default=student_list,  # Default: all students
            key="roster_multiselect"
        )
        
        roster_name = st.text_input("Roster Label:", value=f"{sess['course']} Exam", key="roster_name_input")
        
        if st.button("Set Roster", key="set_roster_btn"):
            # Extract student IDs from "SXXX - Name" format
            candidate_ids = [s.split(" - ")[0] for s in selected_candidates if " - " in s]
            try:
                r = requests.post(f"{API_URL}/exam/roster", json={
                    "exam_session_id": sess["exam_session_id"],
                    "student_ids": candidate_ids,
                    "roster_name": roster_name
                })
                if r.status_code == 200 and r.json()["status"] == "ok":
                    data = r.json()["data"]
                    st.toast(f"Roster set: {data['roster_count']} candidates for {data['roster_name']}", icon="📋")
                    st.session_state.roster_set = True
                else:
                    st.error("Failed to set roster.")
            except Exception as e:
                st.error(f"Roster error: {e}")
        
        if st.session_state.roster_set:
            st.success(f"Roster active: {len(selected_candidates)} candidates")

    st.sidebar.markdown("---")


    # ============================================================
    # MODE SELECTION
    # ============================================================
    st.sidebar.subheader("Mode Selection")
    demo_mode = st.sidebar.toggle("Quick Demo Mode (solo testing)", value=st.session_state.demo_mode)
    st.session_state.demo_mode = demo_mode

    if demo_mode:
        student_list = load_student_list()
        # Find default index for S022
        default_idx = 0
        for i, s in enumerate(student_list):
            if s.startswith("S022"):
                default_idx = i
                break
        demo_student = st.sidebar.selectbox(
            "Active monitored student:",
            options=student_list,
            index=default_idx,
            key="demo_student_selectbox"
        )
        st.session_state.demo_student_selection = demo_student
        active_sid = demo_student.split(" - ")[0]
        st.session_state.selected_student_id = active_sid
        st.sidebar.info(f"Quick Demo Mode active: all events will be attributed to {demo_student}")
    else:
        # Production Mode — Auto-Attribution
        st.sidebar.info("🤖 **Auto-Attribution Active**\nFaces are identified automatically via InsightFace. Events attributed to recognized students.")
        
    st.sidebar.markdown("---")
    
    # DEMO INCIDENT SIMULATORS
    # These inject proctoring events programmatically for live presentation.
    st.sidebar.subheader("Live Demo Incident Injectors")
    
    def log_demo_event(event_type: str):
        """Logs a demo incident event to the backend, using current webcam frame if available."""
        sess_id = st.session_state.exam_session["exam_session_id"]
        stud_id = get_active_student_id()
        
        # Use current webcam frame if available, otherwise generate a synthetic one
        if st.session_state.current_frame_bytes is not None:
            screenshot_bytes = st.session_state.current_frame_bytes
        else:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.rectangle(frame, (100, 100), (540, 380), (0, 0, 255), 4)
            cv2.putText(frame, f"FLAGGED INCIDENT: {event_type.upper()}", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(frame, f"Student: {stud_id} | Session #{sess_id}", (50, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            _, img_encoded = cv2.imencode('.jpg', frame)
            screenshot_bytes = img_encoded.tobytes()
        
        files = {'screenshot': ('event.jpg', screenshot_bytes, 'image/jpeg')}
        data = {
            'exam_session_id': sess_id,
            'student_id': stud_id,
            'event_type': event_type
        }
        
        try:
            r = requests.post(f"{API_URL}/exam/event", data=data, files=files)
            if r.status_code == 200:
                st.toast(f"Logged {event_type} anomaly event for {stud_id}!", icon="🚨")
            else:
                st.error("Error logging incident.")
        except Exception as err:
            st.error(f"Failed to reach backend: {err}")
            
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("📱 Phone (+50)"):
            log_demo_event("phone")
            st.rerun()
        if st.button("📝 Note (+35)"):
            log_demo_event("cheat_sheet")
            st.rerun()
        if st.button("👥 Multi-Face (+45)"):
            log_demo_event("multi_face")
            st.rerun()
    with col2:
        if st.button("📖 Book (+30)"):
            log_demo_event("book")
            st.rerun()
        if st.button("👀 Gaze (>3s) (+20)"):
            log_demo_event("gaze_deviation")
            st.rerun()
        if st.button("❓ Unverified (+60)"):
            log_demo_event("unverified")
            st.rerun()
            
    st.sidebar.markdown("---")
    if st.sidebar.button("🔴 Terminate Session"):
        st.session_state.exam_session = None
        st.session_state.webcam_active = False
        st.session_state.current_frame_bytes = None
        st.session_state.recognized_faces_cache = {}
        st.session_state.per_student_gaze_timer = {}
        st.session_state.unknown_face_count = 0
        st.session_state.verified_student_ids = set()
        st.toast("Proctoring session closed.", icon="🛑")
        st.rerun()

# Main Panel
st.title("🛡️ ExamShield AI Proctoring Dashboard")

if st.session_state.exam_session is None:
    st.info("Welcome to ExamShield AI! Please fill in the details in the left sidebar and click **Initialize Proctoring Session** to activate the exam console.")
else:
    sess = st.session_state.exam_session

    # ============================================================
    # DASHBOARD METRICS BAR (always visible above tabs)
    # ============================================================
    expected_count = sess.get("expected_count", 0)
    verified_count = len(st.session_state.verified_student_ids)
    recognized_count = len(st.session_state.recognized_faces_cache)
    unknown_count = st.session_state.unknown_face_count

    # Try to get live counts from backend
    try:
        entry_r = requests.get(f"{API_URL}/exam/{sess['exam_session_id']}/entry_status", timeout=2)
        if entry_r.status_code == 200:
            entry_data = entry_r.json()["data"]
            expected_count = entry_data["expected_count"]
            verified_count = entry_data["verified_count"]
            st.session_state.verified_student_ids = set(entry_data["verified"])
    except Exception:
        pass

    # Get total incidents
    total_incidents = 0
    try:
        stu_r = requests.get(f"{API_URL}/exam/{sess['exam_session_id']}/students", timeout=2)
        if stu_r.status_code == 200:
            summary = stu_r.json().get("summary", {})
            total_incidents = summary.get("total_incidents", 0)
    except Exception:
        pass

    met1, met2, met3, met4, met5 = st.columns(5)
    met1.metric("📋 Expected", expected_count)
    met2.metric("✅ Verified", verified_count)
    met3.metric("🟢 Recognized", recognized_count)
    met4.metric("🟡 Unknown", unknown_count)
    met5.metric("🔴 Active Incidents", total_incidents)

    # Main dashboard tabs
    tab_entry, tab_live, tab_grid, tab_report = st.tabs([
        "🚪 Entry Verification",
        "🎥 Live Proctoring Stream",
        "📊 Student Risk Grid",
        "📄 Integrity Verification Report"
    ])
    
    # ----------------------------------------------------
    # TAB 1: Entry Verification
    # ----------------------------------------------------
    with tab_entry:
        st.subheader("🚪 Student Entry Verification")
        st.markdown("Capture a photo of each student at the exam hall entrance. The system will identify them and add them to the **Verified Roster** for monitoring.")

        col_cam, col_progress = st.columns([1, 1])

        with col_cam:
            entry_photo = st.camera_input(label="📸 Entry photo — student facing camera", key="entry_cam")
            if entry_photo is not None:
                entry_bytes = entry_photo.getvalue()
                try:
                    files = {"photo": ("entry.jpg", entry_bytes, "image/jpeg")}
                    data = {"exam_session_id": str(sess["exam_session_id"])}
                    r = requests.post(f"{API_URL}/exam/verify_entry", data=data, files=files)
                    if r.status_code == 200:
                        result = r.json()
                        if result["status"] == "ok":
                            d = result["data"]
                            if d["entry_status"] == "new":
                                st.success(f"✅ **{d['name']}** ({d['student_id']}) verified — Confidence: {d['confidence']:.4f}")
                                st.session_state.verified_student_ids.add(d["student_id"])
                            else:
                                st.info(f"ℹ️ **{d['name']}** ({d['student_id']}) already verified (duplicate scan)")
                            st.markdown(f"**Verified:** {d['verified_count']}/{d['expected_count']}")
                        elif result["status"] == "no_match":
                            st.warning(f"⚠️ {result['message']} (confidence: {result.get('confidence', 'N/A')})")
                        else:
                            st.error(result.get("message", "Verification failed."))
                except Exception as e:
                    st.error(f"Entry verification failed: {e}")

        with col_progress:
            st.markdown("### Verification Progress")

            # Fetch entry status from backend
            try:
                r = requests.get(f"{API_URL}/exam/{sess['exam_session_id']}/entry_status")
                if r.status_code == 200:
                    edata = r.json()["data"]
                    v_count = edata["verified_count"]
                    e_count = edata["expected_count"]

                    # Progress bar
                    progress = v_count / e_count if e_count > 0 else 0
                    st.progress(progress, text=f"{v_count}/{e_count} students verified ({progress*100:.0f}%)")

                    # Verified students list
                    if edata["verified_details"]:
                        st.markdown("**✅ Verified Students:**")
                        for vs in edata["verified_details"]:
                            st.markdown(f"- ✅ **{vs['student_id']}** {vs['name']}")

                    # Not verified list (gray)
                    if edata["not_verified_details"]:
                        st.markdown("**⬜ Not Yet Verified:**")
                        for nv in edata["not_verified_details"]:
                            st.markdown(f"- ⬜ <span style='color: #64748B;'>{nv['student_id']} {nv['name']}</span>", unsafe_allow_html=True)
                else:
                    st.warning("Could not fetch entry status.")
            except Exception as e:
                st.error(f"Error loading entry status: {e}")

    # ----------------------------------------------------
    # TAB 2: Live Monitoring Stream
    # ----------------------------------------------------
    with tab_live:
        if st.session_state.demo_mode:
            st.subheader(f"Seat {get_active_student_id()} Real-time Monitoring")
        else:
            st.subheader("🎥 Classroom Real-time Monitoring (Auto-Attribution)")

        col_feed, col_student_status = st.columns([2, 1])
        
        with col_feed:
            st.markdown("<br/>", unsafe_allow_html=True)

            # --------------------------------------------------------
            # CONTINUOUS WEBCAM MONITORING
            # Demo Mode: attributes all events to dropdown student
            # Production Mode: auto-attributes via face recognition
            # --------------------------------------------------------

            # Initialize event deduplication cooldown tracker
            if "event_cooldowns" not in st.session_state:
                st.session_state.event_cooldowns = {}

            EVENT_COOLDOWN_SECONDS = 30

            def should_log_event(event_type: str, student_id: str = "") -> bool:
                """Returns True if enough time has passed since last event of this type for this student."""
                import time as _time
                now = _time.time()
                key = f"{event_type}_{student_id}" if student_id else event_type
                last = st.session_state.event_cooldowns.get(key, 0)
                if now - last >= EVENT_COOLDOWN_SECONDS:
                    st.session_state.event_cooldowns[key] = now
                    return True
                return False

            # Webcam control buttons
            btn_col1, btn_col2, btn_col3 = st.columns(3)
            with btn_col1:
                if st.button("▶️ Start Webcam", key="start_cam", type="primary"):
                    st.session_state.webcam_active = True
            with btn_col2:
                if st.button("⏹️ Stop Webcam", key="stop_cam"):
                    st.session_state.webcam_active = False
            with btn_col3:
                if st.button("📸 Capture Evidence", key="evidence_btn"):
                    if st.session_state.current_frame_bytes is not None:
                        sess_id = sess["exam_session_id"]
                        # Use recognized student or active student for filename
                        stud_id = list(st.session_state.recognized_faces_cache.keys())[0] if st.session_state.recognized_faces_cache else get_active_student_id()
                        screenshot_dir = settings.SCREENSHOTS_DIR / str(sess_id)
                        screenshot_dir.mkdir(parents=True, exist_ok=True)
                        timestamp_str = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                        filename = f"{stud_id}_{timestamp_str}.jpg"
                        save_path = screenshot_dir / filename
                        nparr = np.frombuffer(st.session_state.current_frame_bytes, np.uint8)
                        save_frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        if save_frame is not None:
                            cv2.imwrite(str(save_path), save_frame)
                            st.toast(f"Evidence screenshot saved: {filename}", icon="📸")
                    else:
                        st.warning("No frame captured yet. Start the webcam first.")

            if st.session_state.webcam_active:
                # Import AI engines
                from backend.core.gaze_engine import estimate_head_pose
                from backend.core.yolo_engine import detect_prohibited_objects
                from backend.core.face_engine import (
                    detect_faces,
                    cosine_similarity,
                    recognize_with_upscaling,
                )

                frame_placeholder = st.empty()
                status_placeholder = st.empty()

                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    st.error("Could not open webcam. Check camera permissions and connections.")
                    st.session_state.webcam_active = False
                else:
                    # Load enrolled students for face matching (production mode)
                    enrolled_students = []
                    student_name_map = {}
                    if not st.session_state.demo_mode:
                        try:
                            from backend.database import SessionLocal
                            from backend.models import Student
                            db_session = SessionLocal()
                            enrolled_students = db_session.query(Student).all()
                            student_name_map = {s.student_id: s.name for s in enrolled_students}
                            # Filter to only verified students if available
                            verified_ids = st.session_state.verified_student_ids
                            if verified_ids:
                                enrolled_students = [s for s in enrolled_students if s.student_id in verified_ids]
                            db_session.close()
                        except Exception as e:
                            st.warning(f"Could not load student DB for recognition: {e}")

                    status_placeholder.success("🟢 Live monitoring active — AI processing frames")

                    # Frame processing config
                    RECOGNITION_INTERVAL = 5   # Face recognition every 5th frame
                    AI_INTERVAL = 3             # YOLO + gaze every 3rd frame
                    frame_counter = 0
                    fps_start_time = time.time()
                    fps_frame_count = 0
                    display_fps = 0.0

                    # Cached AI results
                    cached_identities = {}
                    cached_poses = []
                    cached_objects = []
                    cached_unknown_boxes = []

                    # Continuous monitoring loop
                    while st.session_state.webcam_active:
                        ret, frame = cap.read()
                        if not ret:
                            st.warning("Lost webcam frame. Retrying...")
                            continue

                        frame_counter += 1
                        fps_frame_count += 1
                        h_orig, w_orig = frame.shape[:2]

                        # Calculate display FPS
                        elapsed_fps = time.time() - fps_start_time
                        if elapsed_fps >= 1.0:
                            display_fps = fps_frame_count / elapsed_fps
                            fps_frame_count = 0
                            fps_start_time = time.time()

                        # Downscale for AI processing
                        target_w = 640
                        scale = target_w / w_orig
                        small = cv2.resize(frame, (target_w, int(h_orig * scale)))

                        # ==============================================================
                        # DEMO MODE: Original behavior (dropdown student attribution)
                        # ==============================================================
                        if st.session_state.demo_mode:
                            active_student = get_active_student_id()

                            if frame_counter % AI_INTERVAL == 0:
                                cached_poses = estimate_head_pose(small)
                                cached_objects = detect_prohibited_objects(small)

                            # Draw gaze overlays (scale back to original)
                            for pose in cached_poses:
                                sx1, sy1, sx2, sy2 = pose["bbox"]
                                x1, y1, x2, y2 = int(sx1/scale), int(sy1/scale), int(sx2/scale), int(sy2/scale)
                                yaw = pose["yaw"]
                                pitch = pose["pitch"]
                                dev_active = abs(yaw) > settings.GAZE_MAX_YAW or abs(pitch) > settings.GAZE_MAX_PITCH
                                color = (0, 165, 255) if dev_active else (0, 255, 0)
                                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                                cv2.putText(frame, f"Yaw: {yaw:.1f}deg", (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                                cv2.putText(frame, f"Pitch: {pitch:.1f}deg", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                                if dev_active:
                                    if st.session_state.gaze_timer_start is None:
                                        st.session_state.gaze_timer_start = time.time()
                                    else:
                                        elapsed = time.time() - st.session_state.gaze_timer_start
                                        cv2.putText(frame, f"Deviation: {elapsed:.1f}s", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
                                        if elapsed > settings.GAZE_DEVIATION_TIME_LIMIT and should_log_event("gaze_deviation", active_student):
                                            _, img_encoded = cv2.imencode('.jpg', frame)
                                            files = {'screenshot': ('gaze.jpg', img_encoded.tobytes(), 'image/jpeg')}
                                            data = {'exam_session_id': sess['exam_session_id'], 'student_id': active_student, 'event_type': 'gaze_deviation'}
                                            try:
                                                requests.post(f"{API_URL}/exam/event", data=data, files=files)
                                            except Exception:
                                                pass
                                            st.session_state.gaze_timer_start = time.time()
                                else:
                                    st.session_state.gaze_timer_start = None

                            # Object detection overlays (demo mode)
                            for obj in cached_objects:
                                sox1, soy1, sox2, soy2 = obj["bbox"]
                                ox1, oy1, ox2, oy2 = int(sox1/scale), int(soy1/scale), int(sox2/scale), int(soy2/scale)
                                label = obj["label"]
                                conf = obj["confidence"]
                                cv2.rectangle(frame, (ox1, oy1), (ox2, oy2), (0, 0, 255), 2)
                                cv2.putText(frame, f"PROHIBITED: {label.upper()} ({conf*100:.1f}%)", (ox1, oy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                                if should_log_event(label, active_student):
                                    _, obj_enc = cv2.imencode('.jpg', frame)
                                    obj_files = {'screenshot': ('detection.jpg', obj_enc.tobytes(), 'image/jpeg')}
                                    obj_data = {'exam_session_id': sess['exam_session_id'], 'student_id': active_student, 'event_type': label}
                                    try:
                                        requests.post(f"{API_URL}/exam/event", data=obj_data, files=obj_files)
                                    except Exception:
                                        pass

                        # ==============================================================
                        # PRODUCTION MODE: Auto-Attribution via Face Recognition
                        # ==============================================================
                        else:
                            # --- TIER 1: Face Recognition (every 5th frame) ---
                            if frame_counter % RECOGNITION_INTERVAL == 0 and enrolled_students:
                                try:
                                    faces = detect_faces(small, allow_synthetic=False)
                                    new_identities = {}
                                    unknown_count = 0
                                    new_unknown_boxes = []

                                    for face in faces:
                                        fx1, fy1, fx2, fy2 = face["bbox"]
                                        face_width = fx2 - fx1
                                        frame_h, frame_w = frame.shape[:2]
                                        # Clamp bbox to frame bounds for safe ROI cropping
                                        cx1 = max(0, int(fx1)); cy1 = max(0, int(fy1))
                                        cx2 = min(frame_w, int(fx2)); cy2 = min(frame_h, int(fy2))
                                        face_crop = frame[cy1:cy2, cx1:cx2] if (cx2 > cx1 and cy2 > cy1) else None

                                        # Distance awareness: 3-tier face-recognition ladder.
                                        #   face_width <  FACE_MIN_WIDTH              -> TOO FAR (skip)
                                        #   FACE_MIN_WIDTH <= face_width < FACE_RECOVERY_WIDTH -> MARGINAL (upscale then recognize)
                                        #   face_width >= FACE_RECOVERY_WIDTH         -> GOOD (direct recognize)
                                        if face_width < settings.FACE_MIN_WIDTH:
                                            # Too far -- draw gray box later
                                            new_unknown_boxes.append({
                                                "bbox": [int(fx1/scale), int(fy1/scale), int(fx2/scale), int(fy2/scale)],
                                                "label": "TOO FAR"
                                            })
                                            continue

                                        # Use recognize_with_upscaling() -- handles marginal vs good tier internally
                                        best_sid, best_conf, rec_tier = recognize_with_upscaling(
                                            face_crop=face_crop,
                                            enrolled_students=enrolled_students,
                                            face_width=face_width,
                                        )
                                        if rec_tier == "too_far" or best_sid is None or best_conf < settings.EXAM_ENTRY_MATCH_THRESHOLD:
                                            unknown_count += 1
                                            new_unknown_boxes.append({
                                                "bbox": [int(fx1/scale), int(fy1/scale), int(fx2/scale), int(fy2/scale)],
                                                "label": "UNKNOWN"
                                            })
                                            # Log unknown person event with cooldown -- use the dedicated
                                            # /log_unknown_person endpoint so we don't falsely attribute the
                                            # unknown face to whichever verified student happens to be first.
                                            if should_log_event("unknown_person", "GLOBAL"):
                                                _, unk_enc = cv2.imencode('.jpg', frame)
                                                unk_files = {'screenshot': ('unknown.jpg', unk_enc.tobytes(), 'image/jpeg')}
                                                unk_data = {'exam_session_id': sess['exam_session_id']}
                                                try:
                                                    requests.post(f"{API_URL}/exam/log_unknown_person", data=unk_data, files=unk_files)
                                                except Exception:
                                                    pass

                                        else:
                                            new_identities[best_sid] = {
                                                "bbox": [int(fx1/scale), int(fy1/scale), int(fx2/scale), int(fy2/scale)],
                                                "confidence": round(best_conf, 4),
                                                "name": student_name_map.get(best_sid, best_sid),
                                                "quality": rec_tier,
                                            }

                                    cached_identities = new_identities
                                    cached_unknown_boxes = new_unknown_boxes
                                    st.session_state.recognized_faces_cache = cached_identities
                                    st.session_state.unknown_face_count = unknown_count
                                except Exception as e:
                                    pass  # Recognition failed this frame, use cached

                            # --- TIER 2: YOLO + Gaze (every 3rd frame) ---
                            if frame_counter % AI_INTERVAL == 0:
                                try:
                                    cached_poses = estimate_head_pose(small)
                                except Exception:
                                    cached_poses = []
                                try:
                                    cached_objects = detect_prohibited_objects(small)
                                except Exception:
                                    cached_objects = []

                            # --- Draw recognized identity labels ---
                            for sid, info in cached_identities.items():
                                x1, y1, x2, y2 = info["bbox"]
                                conf = info["confidence"]
                                name = info["name"]
                                quality = info.get("quality", "good")

                                if quality == "marginal":
                                    box_color = (0, 255, 255)  # Yellow
                                    label_text = f"{sid} {name} ({conf*100:.0f}%) LOW QUALITY"
                                else:
                                    box_color = (0, 255, 0)  # Green
                                    label_text = f"{sid} {name} ({conf*100:.0f}%)"

                                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                                # Label background
                                label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                                cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), (x1 + label_size[0] + 4, y1), box_color, -1)
                                cv2.putText(frame, label_text, (x1 + 2, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

                            # --- Draw unknown / too-far face labels ---
                            for unk in cached_unknown_boxes:
                                x1, y1, x2, y2 = unk["bbox"]
                                label = unk["label"]
                                if label == "TOO FAR":
                                    box_color = (128, 128, 128)  # Gray
                                else:
                                    box_color = (0, 255, 255)  # Yellow for UNKNOWN
                                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                                cv2.putText(frame, f"⚠ {label}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

                            # --- Gaze overlays with per-student attribution ---
                            for pose in cached_poses:
                                sx1, sy1, sx2, sy2 = pose["bbox"]
                                x1, y1, x2, y2 = int(sx1/scale), int(sy1/scale), int(sx2/scale), int(sy2/scale)
                                yaw = pose["yaw"]
                                pitch = pose["pitch"]
                                dev_active = abs(yaw) > settings.GAZE_MAX_YAW or abs(pitch) > settings.GAZE_MAX_PITCH

                                # Find which recognized student this gaze belongs to
                                attributed_sid = find_nearest_student([x1, y1, x2, y2], cached_identities)

                                if dev_active and attributed_sid:
                                    color = (0, 165, 255)  # Orange
                                    cv2.putText(frame, f"Yaw: {yaw:.1f}", (x1, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                                    cv2.putText(frame, f"Pitch: {pitch:.1f}", (x1, y2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

                                    # Per-student gaze timer
                                    timers = st.session_state.per_student_gaze_timer
                                    if attributed_sid not in timers:
                                        timers[attributed_sid] = {"start": time.time(), "fired": False}
                                    else:
                                        elapsed = time.time() - timers[attributed_sid]["start"]
                                        cv2.putText(frame, f"Gaze Dev: {elapsed:.1f}s", (x1, y2 + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                                        if elapsed > settings.GAZE_DEVIATION_TIME_LIMIT and should_log_event("gaze_deviation", attributed_sid):
                                            _, gaze_enc = cv2.imencode('.jpg', frame)
                                            gaze_files = {'screenshot': ('gaze.jpg', gaze_enc.tobytes(), 'image/jpeg')}
                                            gaze_data = {'exam_session_id': sess['exam_session_id'], 'student_id': attributed_sid, 'event_type': 'gaze_deviation'}
                                            try:
                                                requests.post(f"{API_URL}/exam/event", data=gaze_data, files=gaze_files)
                                            except Exception:
                                                pass
                                            timers[attributed_sid] = {"start": time.time(), "fired": True}
                                elif attributed_sid:
                                    # Reset timer for this student
                                    if attributed_sid in st.session_state.per_student_gaze_timer:
                                        del st.session_state.per_student_gaze_timer[attributed_sid]

                            # --- Object detection with nearest-face attribution ---
                            for obj in cached_objects:
                                sox1, soy1, sox2, soy2 = obj["bbox"]
                                ox1, oy1, ox2, oy2 = int(sox1/scale), int(soy1/scale), int(sox2/scale), int(soy2/scale)
                                label = obj["label"]
                                conf = obj["confidence"]

                                cv2.rectangle(frame, (ox1, oy1), (ox2, oy1), (0, 0, 255), 2)

                                # Attribute to nearest recognized student
                                attributed_sid = find_nearest_student([ox1, oy1, ox2, oy2], cached_identities)
                                attr_label = f" → {attributed_sid}" if attributed_sid else " → ?"
                                cv2.putText(frame, f"PROHIBITED: {label.upper()} ({conf*100:.0f}%){attr_label}", (ox1, oy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                                if attributed_sid and should_log_event(label, attributed_sid):
                                    _, obj_enc = cv2.imencode('.jpg', frame)
                                    obj_files = {'screenshot': ('detection.jpg', obj_enc.tobytes(), 'image/jpeg')}
                                    obj_data = {'exam_session_id': sess['exam_session_id'], 'student_id': attributed_sid, 'event_type': label}
                                    try:
                                        requests.post(f"{API_URL}/exam/event", data=obj_data, files=obj_files)
                                    except Exception:
                                        pass

                        # FPS overlay (both modes)
                        faces_in_frame = len(cached_identities) if not st.session_state.demo_mode else "-"
                        cv2.putText(frame, f"FPS: {display_fps:.0f} | Faces: {faces_in_frame} | Unknown: {st.session_state.unknown_face_count}",
                                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)

                        # Store current annotated frame for evidence screenshots
                        _, encoded_frame = cv2.imencode('.jpg', frame)
                        st.session_state.current_frame_bytes = encoded_frame.tobytes()

                        # Display annotated frame (BGR -> RGB for Streamlit)
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frame_placeholder.image(frame_rgb, use_container_width=True)

                        # Small sleep to prevent CPU hammering
                        time.sleep(0.05)

                    cap.release()
                    status_placeholder.info("⏹️ Webcam stopped.")

            else:
                st.info("Click **Start Webcam** to begin continuous live monitoring. Each frame is auto-processed through the AI detection pipeline.")

                
        with col_student_status:
            st.subheader("Proctoring Timeline")
            
            if st.session_state.demo_mode:
                # Demo mode: show single student
                try:
                    r = requests.get(f"{API_URL}/exam/{sess['exam_session_id']}/students")
                    if r.status_code == 200:
                        students = r.json()["data"]
                        active_student = next((s for s in students if s["student_id"] == get_active_student_id()), None)
                        
                        if active_student:
                            st.markdown(f"### Student: {active_student['name']}")
                            st.markdown(f"**ID:** `{active_student['student_id']}`")
                            status = active_student["risk_status"]
                            color = active_student["color"]
                            st.markdown(f"Risk Status: <span class='risk-badge risk-{color}'>{status}</span>", unsafe_allow_html=True)
                            st.markdown(f"**Integrity Score:** `{active_student['anomaly_score']}` points")
                            st.markdown("---")
                            st.subheader("Event History")
                            timeline = active_student["timeline"]
                            if timeline:
                                for idx, event in enumerate(reversed(timeline)):
                                    event_label = event["event_type"].replace("_", " ").title()
                                    occ_time = datetime.fromisoformat(event["occurred_at"]).strftime("%I:%M:%S %p")
                                    with st.expander(f"🚨 {event_label} (+{event['score_delta']} pts) at {occ_time}"):
                                        if event["screenshot_path"]:
                                            st.image(f"{BROWSER_API_URL}/{event['screenshot_path']}", use_container_width=True)
                                        else:
                                            st.info("No verification screenshot recorded for this event.")
                            else:
                                st.info("No suspicious activities logged for this student.")
                except Exception as e:
                    st.error(f"Could not load status: {e}")
            else:
                # Production mode: show multi-student summary
                st.markdown("### Currently Recognized")

                # Use the session_state cache (set by the webcam loop) -- the local
                # `cached_identities` variable lives inside the webcam's `while` block
                # and is not visible here.
                live_identities = st.session_state.recognized_faces_cache

                if live_identities:
                    for sid, info in live_identities.items():
                        conf = info["confidence"]
                        name = info["name"]
                        st.markdown(f"🟢 **{sid}** {name} — `{conf*100:.0f}%`")
                else:
                    st.info("No faces recognized. Start the webcam.")

                if st.session_state.unknown_face_count > 0:
                    st.markdown(f"🟡 **UNKNOWN** × {st.session_state.unknown_face_count}")

                st.markdown("---")
                st.markdown("### Recent Events")
                
                try:
                    r = requests.get(f"{API_URL}/exam/{sess['exam_session_id']}/students")
                    if r.status_code == 200:
                        all_students = r.json()["data"]
                        # Collect all events, sort by time
                        all_events = []
                        for s in all_students:
                            for ev in s.get("timeline", []):
                                ev["student_name"] = s["name"]
                                ev["student_id_ref"] = s["student_id"]
                                all_events.append(ev)
                        
                        all_events.sort(key=lambda x: x["occurred_at"], reverse=True)
                        
                        if all_events:
                            for ev in all_events[:10]:  # Show last 10 events
                                ev_label = ev["event_type"].replace("_", " ").title()
                                occ_time = datetime.fromisoformat(ev["occurred_at"]).strftime("%I:%M:%S %p")
                                st.markdown(f"🚨 **{ev_label}** (+{ev['score_delta']}) → {ev['student_id_ref']} at {occ_time}")
                        else:
                            st.info("No incidents recorded yet.")
                except Exception as e:
                    st.error(f"Could not load events: {e}")
                
    # ----------------------------------------------------
    # TAB 3: Student Risk Grid
    # ----------------------------------------------------
    with tab_grid:
        st.subheader("Real-time Exam Hall Student Overview")
        
        try:
            r = requests.get(f"{API_URL}/exam/{sess['exam_session_id']}/students")
            if r.status_code == 200:
                resp_json = r.json()
                students_data = resp_json["data"]
                summary = resp_json.get("summary", {})
                
                # Statistics bar
                total_students = len(students_data)
                verified_cnt = sum(1 for s in students_data if s.get("verified", True))
                unverified_cnt = total_students - verified_cnt
                high_risk_cnt = sum(1 for s in students_data if s["color"] == "red")
                caution_cnt = sum(1 for s in students_data if s["color"] == "amber")
                normal_cnt = sum(1 for s in students_data if s["color"] == "green")
                
                stat_col1, stat_col2, stat_col3, stat_col4, stat_col5 = st.columns(5)
                stat_col1.metric("Total Expected", total_students)
                stat_col2.metric("Verified (Present)", verified_cnt)
                stat_col3.metric("High Risk 🔴", high_risk_cnt, delta_color="inverse")
                stat_col4.metric("Caution 🟡", caution_cnt)
                stat_col5.metric("Normal 🟢", normal_cnt)
                
                st.markdown("<br/>", unsafe_allow_html=True)
                
                # Grid view
                grid_cols = st.columns(3)
                
                for idx, student in enumerate(students_data):
                    col_idx = idx % 3
                    with grid_cols[col_idx]:
                        color = student["color"]
                        status = student["risk_status"]
                        is_verified = student.get("verified", True)
                        card_class = "student-card" if is_verified else "student-card-gray"
                        
                        # Determine score color
                        if color == 'red':
                            score_color = '#EF4444'
                        elif color == 'amber':
                            score_color = '#F59E0B'
                        elif color == 'gray':
                            score_color = '#64748B'
                        else:
                            score_color = '#10B981'
                        
                        verified_icon = "✅" if is_verified else "⬜"
                        
                        st.markdown(f"""
                        <div class="{card_class}">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                                <span style="font-weight: 700; font-size: 16px;">{verified_icon} {student['name']}</span>
                                <span class="risk-badge risk-{color}">{status}</span>
                            </div>
                            <div style="font-size: 13px; color: #94A3B8; margin-bottom: 8px;">ID: <code>{student['student_id']}</code> | Section {student['section']}</div>
                            <div style="font-size: 14px; font-weight: 500;">Anomaly Score: <b style="color: {score_color}">{student['anomaly_score']}</b> points</div>
                            <div style="font-size: 11px; color: #64748B; margin-top: 6px;">Logged Events: {student['events_count']} occurrence(s)</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Add quick monitor selector button (only for verified students)
                        if is_verified:
                            if st.button(f"🔎 Monitor {student['student_id']}", key=f"btn_{student['student_id']}"):
                                st.session_state.selected_student_id = student["student_id"]
                                st.toast(f"Monitored seat shifted to {student['student_id']}!")
            else:
                st.error("Failed to load student proctoring grid.")
        except Exception as e:
            st.error(f"Error fetching real-time student grid: {e}")
            
    # ----------------------------------------------------
    # TAB 4: Integrity verification report
    # ----------------------------------------------------
    with tab_report:
        st.subheader("Integrity Verification Report Exporter")
        st.info("Compile the completed proctoring session events, student timelines, and all flagged verification screenshots into a legal, DIU-ready PDF report.")

        if not st.session_state.demo_mode:
            st.markdown("""
            > **Note:** Identity confirmed at exam entry. In-exam events are auto-attributed
            > via face recognition. Events with no recognized face require manual review.
            """)
        
        pdf_export_url = f"{BROWSER_API_URL}/exam/report/{sess['exam_session_id']}"
        
        st.markdown(f"""
        <div style="background: rgba(30,41,59,0.3); border: 1px solid rgba(255,255,255,0.08); border-radius: 16px; padding: 24px; text-align: center;">
            <h3 style="margin-bottom: 16px;">Generate PDF Integrity sheet</h3>
            <p style="color: #94A3B8; font-size: 13px; margin-bottom: 24px; max-width: 500px; margin-left: auto; margin-right: auto;">
                This export will compile student anomaly statistics, risk distributions, specific timestamps for each prohibited item or gaze deviation, along with the raw screenshot captures, programmatically styled inside ReportLab flowables.
            </p>
            <a href="{pdf_export_url}" target="_blank">
                <button style="background: linear-gradient(135deg, #10B981 0%, #059669 100%); color: white; border: none; border-radius: 12px; padding: 14px 28px; font-weight: 600; font-size: 15px; cursor: pointer; box-shadow: 0 4px 15px rgba(16,185,129,0.3); transition: all 0.3s ease;">
                    📥 Download Legal PDF Verification Report
                </button>
            </a>
        </div>
        """, unsafe_allow_html=True)
