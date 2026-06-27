from django.apps import AppConfig


class ChatConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'chat'

    def ready(self) -> None:
        from chat.services.plan_shortcut_router import apply as apply_plan_shortcut_patch
        from chat.services.rag_excerpt_fallback import apply_rag_excerpt_patch
        from chat.services.expense_policy_session_fix import apply as apply_expense_policy_session_fix
        from chat.services.leave_policy_session_fix import apply as apply_leave_policy_session_fix

        apply_plan_shortcut_patch()
        apply_rag_excerpt_patch()
        apply_expense_policy_session_fix()
        apply_leave_policy_session_fix()
