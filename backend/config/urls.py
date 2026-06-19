from django.conf import settings
from django.contrib import admin
from django.urls import include, path
from drf_spectacular.views import (
    SpectacularAPIView,
    SpectacularRedocView,
    SpectacularSwaggerView,
)
from rest_framework.permissions import AllowAny

urlpatterns = [
    path("admin/", admin.site.urls),
    path("api/", include("chat.urls")),
    path("api/", include("knowledge_base.api.urls")),
    path("api/", include("voice.urls")),
]

if settings.ENABLE_API_DOCS:
    _public = {"authentication_classes": [], "permission_classes": [AllowAny]}
    urlpatterns += [
        path(
            "api/schema/",
            SpectacularAPIView.as_view(**_public),
            name="schema",
        ),
        path(
            "api/docs/",
            SpectacularSwaggerView.as_view(url_name="schema", **_public),
            name="swagger-ui",
        ),
        path(
            "api/redoc/",
            SpectacularRedocView.as_view(url_name="schema", **_public),
            name="redoc",
        ),
    ]
