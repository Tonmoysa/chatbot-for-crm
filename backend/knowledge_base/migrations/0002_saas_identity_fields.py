from django.db import migrations, models


def copy_document_company_to_chunks(apps, schema_editor):
    KnowledgeChunk = apps.get_model("knowledge_base", "KnowledgeChunk")
    for chunk in KnowledgeChunk.objects.select_related("document").iterator():
        chunk.company_id = chunk.document.company_id
        chunk.save(update_fields=["company_id"])


class Migration(migrations.Migration):
    dependencies = [
        ("knowledge_base", "0001_initial"),
    ]

    operations = [
        migrations.AddField(
            model_name="knowledgedocument",
            name="company_id",
            field=models.CharField(db_index=True, default="legacy-company", max_length=64),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name="knowledgedocument",
            name="uploaded_by_employee_id",
            field=models.CharField(default="legacy-employee", max_length=64),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name="knowledgechunk",
            name="company_id",
            field=models.CharField(db_index=True, default="legacy-company", max_length=64),
            preserve_default=False,
        ),
        migrations.AlterField(
            model_name="knowledgedocument",
            name="checksum",
            field=models.CharField(blank=True, db_index=True, default="", max_length=64),
        ),
        migrations.RemoveField(
            model_name="knowledgedocument",
            name="uploaded_by",
        ),
        migrations.RunPython(copy_document_company_to_chunks, migrations.RunPython.noop),
        migrations.AddIndex(
            model_name="knowledgedocument",
            index=models.Index(fields=["company_id", "checksum"], name="kb_doc_company_checksum_idx"),
        ),
        migrations.AddIndex(
            model_name="knowledgedocument",
            index=models.Index(fields=["company_id", "status"], name="kb_doc_company_status_idx"),
        ),
        migrations.AddIndex(
            model_name="knowledgechunk",
            index=models.Index(
                fields=["company_id", "document", "chunk_index"],
                name="kb_chunk_company_doc_idx",
            ),
        ),
    ]
