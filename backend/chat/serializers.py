from rest_framework import serializers

from chat.identity import TenantIdentitySerializerMixin

class HrEnvelopeSerializer(serializers.Serializer):
    """Standard API response envelope (OpenAPI / contract)."""

    trace_id = serializers.CharField()
    intent = serializers.CharField(allow_blank=True, required=False)
    entities = serializers.JSONField()
    decision = serializers.JSONField()
    response = serializers.JSONField()
    status = serializers.CharField()
    sources = serializers.ListField(
        child=serializers.DictField(),
        required=False,
        default=list,
    )


class ChatRequestSerializer(TenantIdentitySerializerMixin):
    message = serializers.CharField(max_length=8000)
    # Optional: text extracted from an uploaded receipt/document
    document_text = serializers.CharField(
        max_length=60000, required=False, allow_blank=True, default=""
    )


class DocumentExtractRequestSerializer(TenantIdentitySerializerMixin):
    file = serializers.FileField()


class IntentRequestSerializer(TenantIdentitySerializerMixin):
    message = serializers.CharField(max_length=8000)


class ExtractRequestSerializer(TenantIdentitySerializerMixin):
    message = serializers.CharField(max_length=8000)
    intent = serializers.CharField(max_length=64)


class DecisionRequestSerializer(TenantIdentitySerializerMixin):
    intent = serializers.CharField(max_length=64)
    entities = serializers.JSONField()


class ChatSessionsQuerySerializer(serializers.Serializer):
    company_id = serializers.CharField(max_length=64)
    employee_id = serializers.CharField(max_length=64)
    limit = serializers.IntegerField(required=False, default=30, min_value=1, max_value=50)


class ChatSessionDetailQuerySerializer(serializers.Serializer):
    company_id = serializers.CharField(max_length=64)
    employee_id = serializers.CharField(max_length=64)


class MockCreateSerializer(TenantIdentitySerializerMixin):
    intent = serializers.CharField(max_length=64)
    entities = serializers.JSONField(required=False, default=dict)
    decision = serializers.JSONField(required=False, default=dict)
