from __future__ import annotations

from nova_voice.audio.conversation import ConversationTracker
from nova_voice.capabilities.registry import CapabilityRegistry
from nova_voice.config import Settings
from nova_voice.interpretation.llama_cpp import LlamaCppInterpreter
from nova_voice.interpretation.skills import load_skills
from nova_voice.persistence import TranscriptStore
from nova_voice.persona import Persona
from nova_voice.providers.nova.client import NovaDashboardClient
from nova_voice.providers.nova.provider import NovaProvider
from nova_voice.service import NovaVoiceService


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
    registry = CapabilityRegistry(allowlist={"nova"})
    registry.register(nova_provider)
    interpreter = LlamaCppInterpreter(
        settings.llm_base_url,
        settings.llm_model,
        skills_text=load_skills(settings.skills_path),
        timeout_seconds=settings.llm_timeout_seconds,
    )
    store = TranscriptStore(settings.database_path, settings.retention_hours)
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
        conversations=conversations,
    )
