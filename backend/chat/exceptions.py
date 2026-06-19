import uuid

from django.conf import settings
from rest_framework.views import exception_handler as drf_exception_handler


def hr_exception_handler(exc, context):
    response = drf_exception_handler(exc, context)
    request = context.get("request")
    trace_id = getattr(request, "trace_id", None) or str(uuid.uuid4())

    if response is not None:
        data = response.data
        if isinstance(data, dict) and "detail" in data and len(data) == 1:
            payload = {
                "trace_id": trace_id,
                "intent": "",
                "entities": {},
                "decision": {},
                "response": {
                    "message": str(data["detail"]),
                    "status": "error",
                    "request_id": "",
                },
                "status": "failed",
            }
        else:
            payload = {
                "trace_id": trace_id,
                "intent": "",
                "entities": {},
                "decision": {},
                "response": {
                    "message": "Request error",
                    "status": "error",
                    "request_id": "",
                },
                "status": "failed",
                "error": data if isinstance(data, (dict, list)) else {"detail": str(data)},
            }
        response.data = payload
        response["X-Trace-Id"] = trace_id
        return response

    return response
