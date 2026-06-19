from django.urls import path

from knowledge_base.api import views

app_name = "knowledge_base"

urlpatterns = [
    path("kb/upload-policy/", views.KbUploadPolicyView.as_view(), name="kb-upload-policy"),
]
