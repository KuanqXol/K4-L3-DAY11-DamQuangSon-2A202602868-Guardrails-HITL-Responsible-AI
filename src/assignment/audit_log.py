"""
Assignment 11 — Audit Log starter (TODO).

Records every interaction for forensics. Never blocks by itself —
other layers catch attacks; this layer makes them reviewable.
"""
from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path


def default_audit_log_path() -> str:
    """Always resolve to <repo>/outputs/… (safe when cwd is src/)."""
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "outputs" / "audit_log.json")


class AuditLogPlugin:
    """Framework-agnostic audit logger (wire into ADK callbacks or your pipeline)."""

    def __init__(self):
        self.name = "audit_log"
        self.logs: list[dict] = []
        self._open: dict[str, dict] = {}

    def record_input(self, *, user_id: str, text: str, request_id: str | None = None):
        """Start an audit record and return its correlation ID."""
        correlation_id = request_id or f"req-{uuid.uuid4().hex[:12]}"
        self._open[correlation_id] = {
            "request_id": correlation_id,
            "user_id": user_id,
            "input": text,
            "started_at": utc_now_iso(),
            "started_monotonic": time.perf_counter(),
        }
        return correlation_id

    def record_output(
        self,
        *,
        user_id: str,
        text: str,
        blocked: bool = False,
        layer: str | None = None,
        request_id: str | None = None,
    ):
        """Finish an audit record, including the decision and elapsed time."""
        correlation_id = request_id
        if correlation_id is None:
            # Backward-compatible fallback for callers that correlate by user.
            correlation_id = next(
                (key for key in reversed(self._open) if self._open[key]["user_id"] == user_id),
                f"req-{uuid.uuid4().hex[:12]}",
            )
        pending = self._open.pop(correlation_id, None) or {
            "request_id": correlation_id,
            "user_id": user_id,
            "input": "",
            "started_at": utc_now_iso(),
            "started_monotonic": time.perf_counter(),
        }
        started = pending.pop("started_monotonic")
        pending.update({
            "output": text,
            "blocked": bool(blocked),
            "layer": layer,
            "finished_at": utc_now_iso(),
            "latency_ms": round((time.perf_counter() - started) * 1000, 3),
        })
        self.logs.append(pending)
        return pending

    def export_json(self, filepath: str | None = None):
        """Write logs to disk (JSON array) under repo-root ``outputs/`` by default."""
        path = Path(filepath or default_audit_log_path())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.logs, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return path


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
