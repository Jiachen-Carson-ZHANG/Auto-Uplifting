import pytest
from unittest.mock import MagicMock, patch
from src.llm.backend import LLMBackend, create_backend, Message
from src.llm.providers.anthropic import AnthropicBackend


def test_message_structure():
    msg = Message(role="user", content="hello")
    assert msg.role == "user"
    assert msg.content == "hello"


def test_create_backend_anthropic():
    backend = create_backend(provider="anthropic", model="claude-haiku-4-5-20251001", api_key="test-key")
    assert isinstance(backend, AnthropicBackend)


def test_create_backend_unknown_raises():
    with pytest.raises(ValueError, match="Unknown provider"):
        create_backend(provider="unknown", model="x", api_key="key")


def test_anthropic_backend_complete():
    """Test AnthropicBackend.complete() calls the API and returns string."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="test response")]
    mock_client.messages.create.return_value = mock_response

    backend = AnthropicBackend(model="claude-haiku-4-5-20251001", client=mock_client)
    messages = [Message(role="user", content="say hello")]
    result = backend.complete(messages=messages, temperature=0.3)

    assert result == "test response"
    mock_client.messages.create.assert_called_once()


def test_anthropic_backend_passes_system_message():
    """System message must be passed as system= kwarg, not in messages list."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="ok")]
    mock_client.messages.create.return_value = mock_response

    backend = AnthropicBackend(model="claude-haiku-4-5-20251001", client=mock_client)
    messages = [
        Message(role="system", content="You are a helpful assistant."),
        Message(role="user", content="hello")
    ]
    backend.complete(messages=messages, temperature=0.5)

    call_kwargs = mock_client.messages.create.call_args.kwargs
    assert call_kwargs["system"] == "You are a helpful assistant."
    # system message must NOT be in the messages list passed to API
    api_messages = call_kwargs["messages"]
    assert all(m["role"] != "system" for m in api_messages)


from src.llm.providers.openai import OpenAIBackend


def test_create_backend_openai():
    mock_openai = MagicMock()
    mock_openai.OpenAI.return_value = MagicMock()
    with patch.dict("sys.modules", {"openai": mock_openai}):
        backend = create_backend(provider="openai", model="gpt-4o", api_key="test-key")
    assert isinstance(backend, OpenAIBackend)


def test_openai_backend_complete():
    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "openai response"
    mock_client.chat.completions.create.return_value = MagicMock(choices=[mock_choice])

    backend = OpenAIBackend(model="gpt-4o", client=mock_client)
    messages = [Message(role="user", content="hello")]
    result = backend.complete(messages=messages, temperature=0.5)

    assert result == "openai response"
    mock_client.chat.completions.create.assert_called_once()


def test_openai_backend_passes_all_messages():
    """OpenAI API accepts system messages in the messages list (unlike Anthropic)."""
    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "ok"
    mock_client.chat.completions.create.return_value = MagicMock(choices=[mock_choice])

    backend = OpenAIBackend(model="gpt-4o", client=mock_client)
    messages = [
        Message(role="system", content="You are helpful."),
        Message(role="user", content="hello")
    ]
    backend.complete(messages=messages)

    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    api_messages = call_kwargs["messages"]
    # System message IS in messages list for OpenAI (unlike Anthropic)
    assert any(m["role"] == "system" for m in api_messages)
