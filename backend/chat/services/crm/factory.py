from django.conf import settings

from chat.services.crm.base import CRMAdapter
from chat.services.crm.mock_crm import get_mock_singleton
from chat.services.crm.real_crm import RealCRMAdapter


def get_crm_adapter() -> CRMAdapter:
    if settings.USE_MOCK_CRM:
        return get_mock_singleton()
    return RealCRMAdapter()
