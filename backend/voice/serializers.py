from rest_framework import serializers

from chat.identity import TenantIdentitySerializerMixin


class VoiceTranscribeRequestSerializer(TenantIdentitySerializerMixin):
    file = serializers.FileField()
    language = serializers.CharField(max_length=16, required=False, allow_blank=True)
