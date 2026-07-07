"""
Anthropic SDK Production Tracing
----------------------------------
Patches the Anthropic client so every messages.create() call is automatically
traced. Works for both regular and streaming responses.

Run:
    pip install agentx[anthropic]
    AGENTX_API_KEY=key ANTHROPIC_API_KEY=sk-ant-... python examples/tracing/anthropic_tracing.py
"""

from agentx import AgentX
from agentx.integrations.anthropic import patch_anthropic_client

agentx = AgentX.from_env()


def main():
    try:
        import anthropic
    except ImportError:
        print("Missing dependency. Run: pip install agentx[anthropic]")
        return

    client = anthropic.Anthropic()

    # One-time patch: all subsequent client.messages.create() calls are traced
    patch_anthropic_client(
        client,
        tracer=agentx.tracer,
        name="claude-support-agent",
        metadata={"env": "production"},
        session_id="session-xyz-789",
    )

    print("Running Anthropic agent with AgentX tracing...")

    # Regular (non-streaming) call — traced automatically
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=256,
        messages=[
            {"role": "user", "content": "How do I cancel my subscription?"}
        ],
    )
    print(f"Response: {response.content[0].text}\n")

    # Streaming call — also traced automatically
    with client.messages.stream(
        model="claude-haiku-4-5-20251001",
        max_tokens=256,
        messages=[
            {"role": "user", "content": "What is your refund policy?"}
        ],
    ) as stream:
        full = ""
        for text in stream.text_stream:
            full += text
        print(f"Streamed: {full}\n")

    agentx.tracer.flush(timeout=10)
    print("Done — traces are live in the AgentX dashboard.")


if __name__ == "__main__":
    main()
