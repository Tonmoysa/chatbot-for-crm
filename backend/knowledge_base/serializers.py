from rest_framework import serializers

from chat.identity import TenantIdentitySerializerMixin


class KbPolicyUploadSerializer(TenantIdentitySerializerMixin):
    file = serializers.FileField()
    title = serializers.CharField(max_length=512, required=False, allow_blank=True)
    policy_type = serializers.CharField(max_length=64, required=False, allow_blank=True)
    department = serializers.CharField(max_length=128, required=False, allow_blank=True)


class KbPolicyUploadResponseSerializer(serializers.Serializer):
    document_id = serializers.CharField()
    chunks_created = serializers.IntegerField()
    status = serializers.CharField()
