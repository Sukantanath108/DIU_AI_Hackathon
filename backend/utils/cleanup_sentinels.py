"""
backend.utils.cleanup_sentinels
================================

One-shot maintenance utility for removing UNKNOWN-* sentinel rows that
``exam.py`` creates on the fly inside ``verify_student_entry`` when an
unrecognised face is presented at exam entry.

These rows exist solely to satisfy the foreign-key constraint on
``exam_events.student_id`` so an ``unknown_person`` event can be recorded
without inventing a real student.  They are NOT real students and should
NOT appear in any roster view, attendance count, or analytics export.

The live code paths already filter them out everywhere it matters
(see ``backend/routers/exam.py::_filter_real_students``), but over time
the table can accumulate a sentinel for every past exam session.  This
script gives the operator a safe way to delete them on demand.

Usage:
    python -m backend.utils.cleanup_sentinels            # dry-run (default)
    python -m backend.utils.cleanup_sentinels --apply    # actually delete
    python -m backend.utils.cleanup_sentinels --verbose  # show the rows first

Safety:
    * Only rows whose ``student_id`` starts with ``UNKNOWN`` (case-insensitive)
      are touched.  Real student IDs (``S000``–``S024``) are NEVER deleted.
    * Dry-run by default.  ``--apply`` is required for destructive actions.
    * Events referencing deleted sentinels will have their ``student_id`` set
      to NULL via ``ON DELETE SET NULL`` if the FK was declared that way;
      otherwise the delete will be refused by SQLite.  Either outcome is
      logged so the operator can audit afterwards.
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

# Allow `python -m backend.utils.cleanup_sentinels` from the project root.
# When imported as a module the parent package initialises the path; when run
# directly we fall back to the workspace append.
try:
    from backend.database import SessionLocal, engine
    from backend.models import Student, ExamEvent
except ModuleNotFoundError:  # pragma: no cover - direct-exec path
    import pathlib
    sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent.parent))
    from backend.database import SessionLocal, engine
    from backend.models import Student, ExamEvent


SENTINEL_PREFIX = "UNKNOWN"


def is_sentinel(student_id: Optional[str]) -> bool:
    """True if the ID looks like a session-scoped unknown-person sentinel."""
    if not student_id:
        return False
    return student_id.upper().startswith(SENTINEL_PREFIX)


def find_sentinel_students() -> List[Student]:
    """Return all Student rows whose student_id starts with UNKNOWN-*."""
    db = SessionLocal()
    try:
        return (
            db.query(Student)
            .filter(Student.student_id.like(f"{SENTINEL_PREFIX}%"))
            .order_by(Student.student_id)
            .all()
        )
    finally:
        db.close()


def count_events_for(sentinel_id: str) -> int:
    """Return how many ExamEvent rows reference a given sentinel student_id."""
    db = SessionLocal()
    try:
        return (
            db.query(ExamEvent)
            .filter(ExamEvent.student_id == sentinel_id)
            .count()
        )
    finally:
        db.close()


def purge_unknown_students(apply: bool = False, verbose: bool = False) -> int:
    """
    Delete all UNKNOWN-* sentinel Student rows.

    Args:
        apply: when True, perform the deletes.  When False, only report.
        verbose: when True, also list each sentinel and the number of
            referencing ExamEvent rows.

    Returns:
        Number of sentinel rows that were (or would be) deleted.
    """
    sentinels = find_sentinel_students()
    if not sentinels:
        print("No UNKNOWN-* sentinel students found. Database is clean.")
        return 0

    print(f"Found {len(sentinels)} UNKNOWN-* sentinel student row(s):")
    if verbose:
        for s in sentinels:
            ref_events = count_events_for(s.student_id)
            print(
                f"  - {s.student_id:20s}  name={s.name!r:40s}  "
                f"referenced_by_events={ref_events}"
            )

    if not apply:
        print()
        print("Dry-run: nothing was deleted.  Re-run with --apply to remove them.")
        return len(sentinels)

    db = SessionLocal()
    deleted = 0
    try:
        for s in sentinels:
            sid = s.student_id
            ref_events = count_events_for(sid)
            db.delete(s)
            db.commit()
            deleted += 1
            print(f"  DELETED  {sid}  (was referenced by {ref_events} event(s))")
    except Exception as exc:
        db.rollback()
        print(f"ERROR during deletion — rolled back. Reason: {exc}")
        raise
    finally:
        db.close()

    print(f"\nDone. Removed {deleted} sentinel student row(s).")
    return deleted


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Remove UNKNOWN-* sentinel students from the database."
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually perform the deletes.  Default is a dry-run.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="List each sentinel row and the number of referencing events.",
    )
    args = parser.parse_args(argv)

    purge_unknown_students(apply=args.apply, verbose=args.verbose)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
