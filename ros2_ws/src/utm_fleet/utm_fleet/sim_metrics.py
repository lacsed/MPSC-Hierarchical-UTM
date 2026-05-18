#!/usr/bin/env python3
from __future__ import annotations

import csv
import os
import threading
import time
from pathlib import Path


class CSVMetricLogger:
    def __init__(self, file_name: str, fieldnames):
        run_dir = os.environ.get("UTM_RUN_DIR", "").strip()
        if not run_dir:
            run_id = os.environ.get("UTM_RUN_ID", time.strftime("%Y%m%d_%H%M%S"))
            run_dir = os.path.join(os.path.expanduser("~"), "utm_runs", run_id)

        self.path = Path(run_dir)
        self.path.mkdir(parents=True, exist_ok=True)

        self.file_path = self.path / file_name
        self.fieldnames = list(fieldnames)
        self.lock = threading.RLock()
        self._created = self.file_path.exists() and self.file_path.stat().st_size > 0

    def write(self, **row):
        now = time.time()
        row.setdefault("t_wall", now)

        clean = {k: row.get(k, "") for k in self.fieldnames}

        with self.lock:
            with open(self.file_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                if not self._created:
                    writer.writeheader()
                    self._created = True
                writer.writerow(clean)
