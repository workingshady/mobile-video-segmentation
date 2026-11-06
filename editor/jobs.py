"""Background jobs (import, model load, propagate, export) with progress + cancel.

GPU work is serialized with one lock so two jobs never fight over VRAM.
"""

import threading
import traceback
import uuid

GPU_LOCK = threading.Lock()


class Job:
    def __init__(self, kind, fn):
        self.id = uuid.uuid4().hex[:10]
        self.kind = kind
        self.fn = fn
        self.status = "running"
        self.progress = 0.0
        self.message = ""
        self.result = None
        self.error = None
        self.cancelled = False
        self.stats = {}

    def update(self, progress=None, message=None, **stats):
        if progress is not None:
            self.progress = float(progress)
        if message is not None:
            self.message = message
        self.stats.update(stats)

    def should_stop(self):
        return self.cancelled

    def run(self):
        try:
            with GPU_LOCK:
                self.result = self.fn(self)
            self.status = "cancelled" if self.cancelled else "done"
            self.progress = 1.0 if not self.cancelled else self.progress
        except Exception as e:  # report every failure to the UI instead of killing the thread silently
            traceback.print_exc()
            self.status = "error"
            self.error = f"{type(e).__name__}: {e}"

    def to_dict(self):
        return {
            "id": self.id,
            "kind": self.kind,
            "status": self.status,
            "progress": self.progress,
            "message": self.message,
            "result": self.result,
            "error": self.error,
            "stats": self.stats,
        }


class JobManager:
    def __init__(self):
        self.jobs = {}

    def start(self, kind, fn):
        job = Job(kind, fn)
        self.jobs[job.id] = job
        threading.Thread(target=job.run, daemon=True).start()
        return job

    def get(self, job_id):
        return self.jobs[job_id]

    def active(self):
        return [j.to_dict() for j in self.jobs.values() if j.status == "running"]
