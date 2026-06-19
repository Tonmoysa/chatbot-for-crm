from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class TranscribeResult:
    text: str
    language: str | None = None
    provider: str = ""


class BaseTranscriptionProvider(ABC):
    name: str = "base"

    @abstractmethod
    def transcribe(
        self,
        *,
        filename: str | None,
        content_type: str | None,
        data: bytes,
        language: str | None,
        trace_id: str,
    ) -> TranscribeResult:
        raise NotImplementedError
