# Framework-specific integrations for production tracing.
# Each sub-module is lazily importable so the core SDK has no extra dependencies.
#
#   from agentx.integrations.langchain import AgentXCallbackHandler
#   from agentx.integrations.crewai import AgentXCrewObserver
#   from agentx.integrations.openai_agents import AgentXTracingProcessor
#   from agentx.integrations.anthropic import patch_anthropic_client
