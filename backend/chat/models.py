from django.db import models


class ConversationSession(models.Model):
    company_id = models.CharField(max_length=64, db_index=True)
    session_id = models.CharField(max_length=64, db_index=True)
    employee_id = models.CharField(max_length=64, db_index=True)
    workflow_state = models.JSONField(blank=True, default=dict)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=("company_id", "employee_id", "session_id"),
                name="chat_session_unique_company_employee_session",
            ),
        ]
        indexes = [
            models.Index(fields=("company_id", "employee_id"), name="chat_session_company_emp_idx"),
        ]

    def __str__(self) -> str:
        return f"{self.company_id}:{self.employee_id}:{self.session_id}"


class ConversationTurn(models.Model):
    ROLE_USER = "user"
    ROLE_ASSISTANT = "assistant"
    ROLE_CHOICES = ((ROLE_USER, "user"), (ROLE_ASSISTANT, "assistant"))

    session = models.ForeignKey(
        ConversationSession, on_delete=models.CASCADE, related_name="turns"
    )
    role = models.CharField(max_length=16, choices=ROLE_CHOICES)
    content = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["created_at"]
