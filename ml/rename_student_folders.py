# ---
# CampusAI Suite — Student Folder Rename Script
# Owner: Member 1 (ML Lead)
#
# Renames name-only folders in data/students/ to the S-prefix format
# (SXXX_Name) that the enrollment pipeline and database expect.
# Idempotent — safe to re-run; already-renamed folders are skipped.
# ---

import os
import re
import sys
from pathlib import Path

# Add workspace directory to python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from backend.core.config import settings


def main():
    students_dir = settings.DATA_DIR / "students"

    if not students_dir.exists():
        print(f"ERROR: Students directory not found: {students_dir}")
        sys.exit(1)

    # Collect all subdirectories
    all_dirs = sorted(
        [d for d in students_dir.iterdir() if d.is_dir()],
        key=lambda d: d.name.lower()
    )

    if not all_dirs:
        print("ERROR: No student folders found in data/students/")
        sys.exit(1)

    # Filter out folders that already match the SXXX_ pattern
    s_prefix_pattern = re.compile(r"^S\d{3}_")
    folders_to_rename = [
        d for d in all_dirs if not s_prefix_pattern.match(d.name)
    ]
    already_renamed = [
        d for d in all_dirs if s_prefix_pattern.match(d.name)
    ]

    if already_renamed:
        print(f"\nAlready renamed ({len(already_renamed)} folders):")
        for d in already_renamed:
            print(f"  {d.name}")

    if not folders_to_rename:
        print("\nAll folders are already in SXXX_Name format. Nothing to rename.")
        # Still write the mapping file from all existing folders
        write_mapping_file(students_dir, sorted(
            [d for d in students_dir.iterdir() if d.is_dir()],
            key=lambda d: d.name.lower()
        ))
        sys.exit(0)

    # Sort alphabetically (case-insensitive) for ID assignment
    folders_to_rename.sort(key=lambda d: d.name.lower())

    # Determine starting ID — account for already-renamed folders
    start_id = len(already_renamed)

    # Build the rename plan
    rename_plan = []
    for idx, folder in enumerate(folders_to_rename):
        new_id = f"S{start_id + idx:03d}"
        new_name = f"{new_id}_{folder.name}"
        rename_plan.append((folder, new_name))

    # Print preview table
    print(f"\n{'='*60}")
    print(f"  FOLDER RENAME PREVIEW — {len(rename_plan)} folders")
    print(f"{'='*60}")
    print(f"  {'#':<5} {'Current Name':<35} → {'New Name'}")
    print(f"  {'-'*5} {'-'*35}   {'-'*40}")

    for idx, (folder, new_name) in enumerate(rename_plan):
        print(f"  {idx:<5} {folder.name:<35} → {new_name}")

    print(f"{'='*60}\n")

    # Ask for confirmation
    confirm = input("Proceed with rename? (y/n): ").strip().lower()
    if confirm != "y":
        print("Aborted. No folders were renamed.")
        sys.exit(0)

    # Perform renames
    print()
    for folder, new_name in rename_plan:
        new_path = folder.parent / new_name
        try:
            folder.rename(new_path)
            print(f"  ✓ {folder.name} → {new_name}")
        except Exception as e:
            print(f"  ✗ FAILED to rename {folder.name}: {e}")

    print(f"\nRenamed {len(rename_plan)} folders successfully.\n")

    # Write the mapping file from all folders (now renamed)
    all_final_dirs = sorted(
        [d for d in students_dir.iterdir() if d.is_dir()],
        key=lambda d: d.name.lower()
    )
    write_mapping_file(students_dir, all_final_dirs)


def write_mapping_file(students_dir: Path, all_dirs: list):
    """Writes data/student_id_mapping.txt with the final ID → name mapping."""
    mapping_path = settings.DATA_DIR / "student_id_mapping.txt"

    # Find the developer student (Sukanta_nath)
    developer_line = None
    lines = []
    for d in all_dirs:
        parts = d.name.split("_", 1)
        if len(parts) >= 2:
            sid = parts[0]
            name = parts[1]
            lines.append(f"{sid} | {name}")
            if "sukanta" in name.lower():
                developer_line = f"# Developer test student: {sid} ({name})"

    with open(mapping_path, "w", encoding="utf-8") as f:
        if developer_line:
            f.write(developer_line + "\n")
        for line in lines:
            f.write(line + "\n")

    print(f"Mapping saved to {mapping_path}")
    print(f"Run next: python ml/enroll_batch.py to populate the database.")


if __name__ == "__main__":
    main()
