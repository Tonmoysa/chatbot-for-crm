from django.urls import path

from voice import views

app_name = "voice"

urlpatterns = [
    path("voice/transcribe/", views.VoiceTranscribeView.as_view(), name="transcribe"),
]
