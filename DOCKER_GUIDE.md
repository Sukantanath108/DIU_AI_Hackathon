# Docker Workflow Guide — CampusAI Suite

## Docker Compose Service Names (source of truth)

These are the actual service names defined in `docker-compose.yml`:

| Service Name           | Port  | Description                          |
|------------------------|-------|--------------------------------------|
| `backend`              | 8000  | FastAPI backend API                  |
| `smartattend-frontend`  | 3000  | React PWA for attendance             |
| `examshield-frontend`   | 8501  | Streamlit proctoring dashboard       |

Always use these exact names in all `docker compose` commands.

## Prerequisites
Docker Desktop must be running before any commands below will work.
Check: the Docker icon in the Windows system tray should show as green
or "running". If it is not running, launch Docker Desktop and wait for
it to finish starting (approximately 30-60 seconds).

## First-time setup (only needed once after cloning the repo)
```powershell
docker compose up --build
```
This builds all 3 container images and starts the services.
Takes approximately 3-5 minutes. Only run with --build when you have
changed code or on first-time setup.

## Normal daily startup
```powershell
docker compose up
```
Starts all containers using already-built images. Takes ~10 seconds.
Run this every time you restart your PC and want to use the system.

## Stopping everything
```powershell
docker compose down
```

## Restarting one service after a code change (faster than full rebuild)
```powershell
docker compose restart backend              # after any change in backend/
docker compose restart examshield-frontend   # after any change in examshield-frontend/
docker compose restart smartattend-frontend  # after any change in smartattend-frontend/
```

## Checking which services are running
```powershell
docker compose ps
```

## Viewing live logs from a specific service
```powershell
docker compose logs -f backend
docker compose logs -f examshield-frontend
docker compose logs -f smartattend-frontend
```
(Press Ctrl+C to stop following logs.)

## Webcam limitation on Windows (IMPORTANT)

Docker Desktop on Windows does **not** support USB webcam passthrough.
Any feature requiring a live webcam (ExamShield real-time proctoring)
must run directly with Python outside Docker.

### Recommended split-mode setup for webcam testing:

```powershell
# 1. Start ONLY backend + SmartAttend in Docker:
docker compose up backend smartattend-frontend

# 2. Stop the Docker ExamShield container (frees port 8501):
docker compose stop examshield-frontend

# 3. Run ExamShield directly on Windows for webcam access:
cd examshield-frontend
.\run_direct.ps1
```

The direct-run script defaults to **port 8601** to avoid conflicting
with the Docker ExamShield service on port 8501. You can override:
```powershell
$env:STREAMLIT_PORT = "8503"; .\run_direct.ps1
```

### Why this split works:
- **backend** in Docker: fine, it doesn't need webcam access
- **smartattend-frontend** in Docker: fine, the browser handles camera
  access via `getUserMedia`, not the container
- **examshield-frontend** direct: REQUIRED on Windows for webcam,
  because Streamlit's `st.camera_input` needs the browser's webcam,
  and the AI engines (YOLO, MediaPipe) run inside the same Python process

## Service URLs
| Service | URL |
|---|---|
| SmartAttend React PWA | http://localhost:3000 |
| ExamShield (Docker) | http://localhost:8501 |
| ExamShield (direct-run) | http://localhost:8601 |
| FastAPI backend docs | http://localhost:8000/docs |

The `/docs` endpoint provides interactive API documentation where you can
test any endpoint directly in the browser — use this to debug backend
calls without needing the frontend.
