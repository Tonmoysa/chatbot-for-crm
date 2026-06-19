from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("chat", "0002_conversationsession_workflow_state"),
    ]

    operations = [
        migrations.AddField(
            model_name="conversationsession",
            name="company_id",
            field=models.CharField(db_index=True, default="legacy-company", max_length=64),
            preserve_default=False,
        ),
        migrations.AlterField(
            model_name="conversationsession",
            name="session_id",
            field=models.CharField(db_index=True, max_length=64),
        ),
        migrations.AlterField(
            model_name="conversationsession",
            name="employee_id",
            field=models.CharField(db_index=True, default="legacy-employee", max_length=64),
            preserve_default=False,
        ),
        migrations.AddConstraint(
            model_name="conversationsession",
            constraint=models.UniqueConstraint(
                fields=("company_id", "employee_id", "session_id"),
                name="chat_session_unique_company_employee_session",
            ),
        ),
        migrations.AddIndex(
            model_name="conversationsession",
            index=models.Index(fields=["company_id", "employee_id"], name="chat_session_company_emp_idx"),
        ),
    ]
