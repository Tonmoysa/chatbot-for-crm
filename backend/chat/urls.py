from django.urls import path

from chat import views

app_name = "chat"

urlpatterns = [
    path("health/", views.HealthView.as_view(), name="health"),
    path("chat/", views.ChatView.as_view(), name="chat"),
    path("chat/sessions/", views.ChatSessionsListView.as_view(), name="chat-sessions"),
    path(
        "chat/sessions/<str:session_id>/",
        views.ChatSessionDetailView.as_view(),
        name="chat-session-detail",
    ),
    path("document/extract/", views.DocumentExtractView.as_view(), name="document-extract"),
    path("intent/", views.IntentView.as_view(), name="intent"),
    path("extract/", views.ExtractView.as_view(), name="extract"),
    path("decision/", views.DecisionView.as_view(), name="decision"),
    path("status/<str:request_id>/", views.RequestStatusView.as_view(), name="status"),
    path("mock/request-create/", views.MockRequestCreateView.as_view(), name="mock-create"),
    path("mock/request-status/", views.MockRequestStatusView.as_view(), name="mock-status"),
]
