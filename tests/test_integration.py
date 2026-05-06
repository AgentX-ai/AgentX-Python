"""
Integration tests for the AgentX Python SDK.

Requires a real API key — set AGENTX_API_KEY in env (or a local .env file).
Tests are skipped automatically if the key is absent.

Run:
    pytest tests/test_integration.py -v
    pytest tests/test_integration.py -v -k "not chat"   # skip slow streaming tests
"""
import os
import pytest

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

API_KEY = os.getenv("AGENTX_API_KEY")
pytestmark = pytest.mark.skipif(not API_KEY, reason="AGENTX_API_KEY not set")


@pytest.fixture(scope="session")
def client():
    from agentx import AgentX
    return AgentX.from_env()


@pytest.fixture(scope="session")
def agents(client):
    return client.list_agents()


@pytest.fixture(scope="session")
def first_agent(agents):
    if not agents:
        pytest.skip("No agents available in this account")
    return agents[0]


@pytest.fixture(scope="session")
def workforces(client):
    return client.list_workforces()


# ---------------------------------------------------------------------------
# Profile
# ---------------------------------------------------------------------------

def test_get_profile(client):
    profile = client.get_profile()
    assert isinstance(profile, dict)
    assert "email" in profile or "_id" in profile


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------

def test_list_agents_returns_list(agents):
    assert isinstance(agents, list)


def test_list_agents_have_required_fields(agents):
    for agent in agents:
        assert agent.id
        assert agent.name


def test_get_agent_by_id(client, first_agent):
    fetched = client.get_agent(first_agent.id)
    assert fetched.id == first_agent.id
    assert fetched.name == first_agent.name


def test_get_agent_invalid_id(client):
    with pytest.raises(Exception):
        client.get_agent("000000000000000000000000")


# ---------------------------------------------------------------------------
# Agent conversations
# ---------------------------------------------------------------------------

def test_list_agent_conversations(first_agent):
    convs = first_agent.list_conversations()
    assert isinstance(convs, list)


def test_new_agent_conversation(first_agent):
    conv = first_agent.new_conversation()
    assert conv.id
    assert conv.agent_id == first_agent.id


def test_agent_conversation_list_messages(first_agent):
    convs = first_agent.list_conversations()
    if not convs:
        pytest.skip("No existing conversations for this agent")
    conv_with_messages = None
    for conv in convs:
        try:
            msgs = conv.list_messages()
            if msgs:
                conv_with_messages = (conv, msgs)
                break
        except Exception:
            continue
    if conv_with_messages is None:
        pytest.skip("No conversations with messages found")
    conv, msgs = conv_with_messages
    assert isinstance(msgs, list)
    for msg in msgs:
        assert msg.id
        assert msg.role in ("user", "bot")


@pytest.mark.slow
def test_agent_chat(first_agent):
    conv = first_agent.new_conversation()
    response = conv.chat("Hello, please reply with just the word PONG.")
    assert response is not None


@pytest.mark.slow
def test_agent_chat_stream(first_agent):
    conv = first_agent.new_conversation()
    chunks = list(conv.chat_stream("Reply with a single word: PONG."))
    assert len(chunks) > 0
    full_text = "".join(c.text or "" for c in chunks)
    assert len(full_text) > 0


# ---------------------------------------------------------------------------
# Workforces
# ---------------------------------------------------------------------------

def test_list_workforces_returns_list(workforces):
    assert isinstance(workforces, list)


def test_workforce_fields(workforces):
    for wf in workforces:
        assert wf.id
        assert wf.name
        assert wf.manager is not None
        assert isinstance(wf.agents, list)


def test_workforce_list_conversations(workforces):
    if not workforces:
        pytest.skip("No workforces available")
    wf = workforces[0]
    convs = wf.list_conversations()
    assert isinstance(convs, list)


def test_workforce_new_conversation(workforces):
    if not workforces:
        pytest.skip("No workforces available")
    wf = workforces[0]
    conv = wf.new_conversation()
    assert conv.id


@pytest.mark.slow
def test_workforce_chat_stream(workforces):
    if not workforces:
        pytest.skip("No workforces available")
    wf = workforces[0]
    conv = wf.new_conversation()
    chunks = list(wf.chat_stream(conv.id, "Reply with a single word: PONG."))
    assert len(chunks) > 0
