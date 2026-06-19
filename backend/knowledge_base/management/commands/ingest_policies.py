import os
from pathlib import Path

from django.core.management.base import BaseCommand

from chat.services.observability import log_step
from knowledge_base.services.ingest import ingest_path


class Command(BaseCommand):
    help = "Bulk-ingest HR policy documents from configured directories into Qdrant."

    def add_arguments(self, parser) -> None:
        parser.add_argument(
            "--dir",
            action="append",
            dest="dirs",
            help="Directory to scan (repeatable). Defaults to KB_POLICY_DIRS env.",
        )
        parser.add_argument(
            "--reindex",
            action="store_true",
            help="Force re-index even when checksum matches an indexed document.",
        )
        parser.add_argument(
            "--trace-id",
            default="cli-ingest",
            help="Trace id for structured logs.",
        )
        parser.add_argument("--company-id", required=True, help="Tenant/company id.")
        parser.add_argument(
            "--employee-id",
            required=True,
            help="Uploader employee id from the SaaS CRM identity model.",
        )

    def handle(self, *args, **options):
        trace_id = str(options.get("trace_id") or "cli-ingest")
        dirs = options.get("dirs") or []
        if not dirs:
            raw = (os.environ.get("KB_POLICY_DIRS") or "").strip()
            dirs = [d.strip() for d in raw.split(",") if d.strip()]
        if not dirs:
            self.stderr.write(
                "No directories: set KB_POLICY_DIRS or pass --dir path(s)."
            )
            return

        reindex = bool(options.get("reindex"))
        company_id = str(options["company_id"]).strip()
        employee_id = str(options["employee_id"]).strip()
        exts = {".md", ".markdown", ".txt", ".pdf", ".png", ".jpg", ".jpeg", ".webp"}
        total = 0
        for d in dirs:
            root = Path(d)
            if not root.is_dir():
                self.stderr.write(f"Skip (not a directory): {root}")
                continue
            for path in sorted(root.rglob("*")):
                if not path.is_file():
                    continue
                if path.suffix.lower() not in exts:
                    continue
                log_step(trace_id, "kb_ingest_file_start", {"path": str(path)})
                r = ingest_path(
                    path,
                    trace_id=trace_id,
                    reindex=reindex,
                    metadata={},
                    company_id=company_id,
                    uploaded_by_employee_id=employee_id,
                )
                self.stdout.write(f"{path}: {r}")
                total += 1
        log_step(trace_id, "kb_ingest_command_done", {"files": total})
        self.stdout.write(self.style.SUCCESS(f"Processed {total} file(s)."))
