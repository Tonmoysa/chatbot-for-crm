# Generated manually for multi-step leave workflow

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("chat", "0001_initial"),
    ]

    operations = [
        migrations.AddField(
            model_name="conversationsession",
            name="workflow_state",
            field=models.JSONField(blank=True, default=dict),
        ),
    ]
