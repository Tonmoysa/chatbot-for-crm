import uuid

from django.utils.deprecation import MiddlewareMixin


class TraceIdMiddleware(MiddlewareMixin):
    """Attach trace_id to request for observability."""

    def process_request(self, request):
        tid = request.META.get("HTTP_X_TRACE_ID") or str(uuid.uuid4())
        request.trace_id = tid

    def process_response(self, request, response):
        if hasattr(request, "trace_id"):
            response["X-Trace-Id"] = request.trace_id
        return response
