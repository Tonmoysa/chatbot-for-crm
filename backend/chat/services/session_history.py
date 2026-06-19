"""List and load persisted chat sessions (database-backed)."""

from __future__ import annotations

from django.db.models import Count, Prefetch

from chat.models import ConversationSession, ConversationTurn

_TITLE_MAX = 56


def _title_from_content(text: str) -> str:
    cleaned = " ".join((text or "").split())
    if not cleaned:
        return "New conversation"
    if len(cleaned) <= _TITLE_MAX:
        return cleaned
    return cleaned[: _TITLE_MAX - 1].rstrip() + "…"


def list_sessions(
    *,
    company_id: str,
    employee_id: str,
    limit: int = 30,
) -> list[dict]:
    company = (company_id or "").strip()
    emp = (employee_id or "").strip()
    if not company or not emp:
        return []

    cap = max(1, min(int(limit or 30), 50))
    user_turns = Prefetch(
        "turns",
        queryset=ConversationTurn.objects.filter(role=ConversationTurn.ROLE_USER).order_by(
            "created_at"
        ),
        to_attr="user_turns",
    )
    qs = (
        ConversationSession.objects.filter(company_id=company, employee_id=emp)
        .annotate(turn_count=Count("turns"))
        .filter(turn_count__gt=0)
        .prefetch_related(user_turns)
        .order_by("-updated_at")[:cap]
    )

    out: list[dict] = []
    for session in qs:
        first = (getattr(session, "user_turns", None) or [None])[0]
        title = _title_from_content(first.content if first else "")
        last = (
            session.turns.order_by("-created_at").values_list("content", flat=True).first()
        )
        out.append(
            {
                "session_id": session.session_id,
                "title": title,
                "updated_at": session.updated_at.isoformat(),
                "preview": _title_from_content(last or "") if last else "",
            }
        )
    return out


def get_session_messages(
    *,
    company_id: str,
    employee_id: str,
    session_id: str,
) -> list[dict] | None:
    company = (company_id or "").strip()
    emp = (employee_id or "").strip()
    sid = (session_id or "").strip()
    if not company or not emp or not sid:
        return None

    try:
        session = ConversationSession.objects.get(
            company_id=company,
            employee_id=emp,
            session_id=sid,
        )
    except ConversationSession.DoesNotExist:
        return None

    return [
        {"role": turn.role, "content": turn.content}
        for turn in session.turns.order_by("created_at")
    ]
