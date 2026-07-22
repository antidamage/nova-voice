from __future__ import annotations

from nova_voice.audio.conversation import ConversationTracker
from nova_voice.authority import HouseholdAuthority
from nova_voice.automation import AutomationManager
from nova_voice.capabilities.registry import CapabilityRegistry
from nova_voice.config import Settings
from nova_voice.durable.store import DurableAgentStore
from nova_voice.events import HouseholdEventConsumer
from nova_voice.interpretation.llama_cpp import LlamaCppInterpreter
from nova_voice.interpretation.skills import load_skills
from nova_voice.memory import MemPalaceClient
from nova_voice.persistence import TranscriptStore
from nova_voice.persona import Persona
from nova_voice.proactive import ProactiveInterventionEngine
from nova_voice.providers.icloud.client import ICloudCalDAVClient
from nova_voice.providers.icloud.provider import ICloudProvider
from nova_voice.providers.nova.client import NovaDashboardClient
from nova_voice.providers.nova.provider import NovaProvider
from nova_voice.providers.personal.provider import PersonalDataProvider
from nova_voice.providers.personal.store import PersonalDataStore
from nova_voice.providers.web.client import BraveScrapeClient, GeminiClient, WebSearchClient
from nova_voice.providers.web.provider import WebProvider
from nova_voice.service import NovaVoiceService
from nova_voice.speaker_profiles import SpeakerProfileStore


def build_service(settings: Settings) -> NovaVoiceService:
    nova_client = NovaDashboardClient(
        settings.nova_base_url,
        mcp_token=settings.nova_mcp_token,
    )
    nova_provider = NovaProvider(
        nova_client,
        contract_version=settings.nova_contract_version,
        alias_refresh_seconds=settings.alias_refresh_seconds,
    )
    gemini_client = (
        GeminiClient(
            settings.web_gemini_api_key,
            settings.web_gemini_model,
            settings.web_gemini_base_url,
            timeout_seconds=settings.web_request_timeout_seconds,
        )
        if settings.web_gemini_api_key
        else None
    )
    web_provider = WebProvider(
        gemini=gemini_client,
        search=WebSearchClient(
            results=settings.web_search_results,
            fetch_max_bytes=settings.web_fetch_max_bytes,
            max_result_chars=settings.web_max_result_chars,
            timeout_seconds=settings.web_request_timeout_seconds,
        ),
        brave=BraveScrapeClient(
            settings.web_search_service_url,
            timeout_seconds=settings.web_search_service_timeout_seconds,
        ),
        default_backend=settings.web_backend_default,
        search_results=settings.web_search_results,
    )
    registry = CapabilityRegistry(allowlist={"nova", "web", "icloud", "personal"})
    registry.register(nova_provider)
    registry.register(web_provider)
    registry.register(PersonalDataProvider(PersonalDataStore(settings.personal_data_path)))
    if settings.icloud_configured:
        registry.register(
            ICloudProvider(
                ICloudCalDAVClient(
                    username=settings.icloud_username or "",
                    app_password=settings.icloud_app_password or "",
                    calendar_url=settings.icloud_calendar_url or "",
                    reminders_url=settings.icloud_reminders_url or "",
                    timeout_seconds=settings.icloud_timeout_seconds,
                )
            )
        )
    interpreter = LlamaCppInterpreter(
        settings.llm_base_url,
        settings.llm_model,
        skills_text=load_skills(settings.skills_path),
        timeout_seconds=settings.llm_timeout_seconds,
    )
    store = TranscriptStore(settings.database_path, settings.retention_hours)
    durable_store = DurableAgentStore(settings.effective_durable_database_path)
    authority = HouseholdAuthority(durable_store, settings.household_tzinfo)
    automations = AutomationManager(durable_store)
    proactive = ProactiveInterventionEngine(durable_store, automations=automations)
    memory = MemPalaceClient(
        settings.mempalace_url,
        settings.mempalace_token if settings.mempalace_enabled else None,
        timeout_seconds=settings.mempalace_timeout_seconds,
    )
    event_consumer = HouseholdEventConsumer(
        nova_client,
        durable_store,
        poll_seconds=settings.household_event_poll_seconds,
        batch_size=settings.household_event_batch_size,
        retention_days=settings.household_event_retention_days,
        on_event=proactive.handle_event,
    )
    speaker_profiles = SpeakerProfileStore(
        settings.database_path,
        retention_days=settings.speaker_candidate_retention_days,
        activation_samples=settings.speaker_activation_samples,
        match_threshold=settings.speaker_match_threshold,
        match_margin=settings.speaker_match_margin,
        cluster_threshold=settings.speaker_cluster_threshold,
    )
    persona = Persona.load(settings.persona_path)
    conversations = ConversationTracker(
        idle_seconds=settings.conversation_idle_seconds,
        max_seconds=settings.conversation_max_seconds,
    )
    return NovaVoiceService(
        settings,
        interpreter,
        registry,
        nova_provider,
        store,
        persona,
        speaker_profiles=speaker_profiles,
        conversations=conversations,
        web_provider=web_provider,
        durable_store=durable_store,
        event_consumer=event_consumer,
        authority=authority,
        automations=automations,
        proactive=proactive,
        memory=memory,
    )
