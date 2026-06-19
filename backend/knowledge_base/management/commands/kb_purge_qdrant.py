"""
Remove stale vectors from Qdrant when Django rows were deleted without signals.

Examples:
  python manage.py kb_purge_qdrant --company-id acme
  python manage.py kb_purge_qdrant --all-companies
"""

from __future__ import annotations

import uuid

from django.core.management.base import BaseCommand, CommandError

from knowledge_base.models import KnowledgeDocument
from knowledge_base.services.qdrant_service import (
    collection_name,
    delete_by_document_id,
    get_qdrant_client,
    purge_company_vectors,
)


class Command(BaseCommand):
    help = "Delete Qdrant policy vectors (by company or all known document ids)."

    def add_arguments(self, parser):
        parser.add_argument(
            "--company-id",
            dest="company_id",
            default="",
            help="Purge all vectors for this tenant (use after bulk admin deletes).",
        )
        parser.add_argument(
            "--all-companies",
            action="store_true",
            help="Purge vectors for every company_id that still has KnowledgeDocument rows "
            "(plus --orphans scan).",
        )
        parser.add_argument(
            "--orphans-only",
            action="store_true",
            help="Delete Qdrant points whose document_db_id is not in Django (scroll collection).",
        )

    def handle(self, *args, **options):
        company_id = (options.get("company_id") or "").strip()
        all_companies = bool(options.get("all_companies"))
        orphans_only = bool(options.get("orphans_only"))

        if not company_id and not all_companies and not orphans_only:
            raise CommandError(
                "Specify --company-id, --all-companies, or --orphans-only."
            )

        trace_id = f"kb-purge-{uuid.uuid4().hex[:10]}"

        if company_id:
            n = purge_company_vectors(company_id, trace_id=trace_id)
            self.stdout.write(
                self.style.SUCCESS(
                    f"Purged Qdrant vectors for company_id={company_id!r} "
                    f"(collection={collection_name()}, trace_id={trace_id})."
                )
            )
            if n is not None:
                self.stdout.write(f"Delete operation reported affected points: {n}")
            return

        if all_companies:
            company_ids = list(
                KnowledgeDocument.objects.values_list("company_id", flat=True).distinct()
            )
            for cid in company_ids:
                if not (cid or "").strip():
                    continue
                purge_company_vectors(str(cid).strip(), trace_id=trace_id)
            self.stdout.write(
                self.style.SUCCESS(
                    f"Re-synced Qdrant deletes for {len(company_ids)} companies "
                    f"with Django documents."
                )
            )

        if orphans_only or all_companies:
            removed = self._purge_orphan_points(trace_id)
            self.stdout.write(
                self.style.SUCCESS(f"Removed {removed} orphan Qdrant point(s).")
            )

    def _purge_orphan_points(self, trace_id: str) -> int:
        valid_ids = set(KnowledgeDocument.objects.values_list("pk", flat=True))
        client = get_qdrant_client()
        name = collection_name()
        removed = 0
        offset = None
        while True:
            points, offset = client.scroll(
                collection_name=name,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            if not points:
                break
            for pt in points:
                payload = getattr(pt, "payload", None) or {}
                doc_id = payload.get("document_db_id")
                if doc_id is None:
                    continue
                try:
                    doc_pk = int(doc_id)
                except (TypeError, ValueError):
                    continue
                if doc_pk not in valid_ids:
                    company = str(payload.get("company_id") or "").strip()
                    if company:
                        delete_by_document_id(doc_pk, company_id=company, trace_id=trace_id)
                    else:
                        from qdrant_client.models import PointIdsList

                        client.delete(
                            collection_name=name,
                            points_selector=PointIdsList(points=[str(pt.id)]),
                        )
                    removed += 1
            if offset is None:
                break
        return removed
