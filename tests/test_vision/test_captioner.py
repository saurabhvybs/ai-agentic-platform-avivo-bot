import pytest
from unittest.mock import AsyncMock, MagicMock
import openai
from vision.captioner import ImageCaptioner
from shared.models import VisionResult


@pytest.fixture
def mock_openai_client():
    return MagicMock(spec=openai.AsyncOpenAI)


@pytest.fixture
def captioner(mock_openai_client):
    return ImageCaptioner(mock_openai_client, max_size_mb=20)


def _mock_response(content: str):
    response = MagicMock()
    response.choices[0].message.content = content
    return response


async def test_describe_returns_vision_result(captioner, mock_openai_client):
    mock_openai_client.chat.completions.create = AsyncMock(
        return_value=_mock_response(
            "A dog running in a park. Tags: dog, park, running"
        )
    )
    result = await captioner.describe(b"fake image bytes")
    assert isinstance(result, VisionResult)
    assert "dog" in result.caption.lower()
    assert "dog" in result.tags


async def test_describe_parses_tags_correctly(captioner, mock_openai_client):
    mock_openai_client.chat.completions.create = AsyncMock(
        return_value=_mock_response("Nice sunset photo. Tags: sunset, sky, orange")
    )
    result = await captioner.describe(b"bytes")
    assert result.tags == ["sunset", "sky", "orange"]


async def test_describe_limits_to_3_tags(captioner, mock_openai_client):
    mock_openai_client.chat.completions.create = AsyncMock(
        return_value=_mock_response("Caption. Tags: a, b, c, d, e")
    )
    result = await captioner.describe(b"bytes")
    assert len(result.tags) == 3


async def test_describe_handles_missing_tags(captioner, mock_openai_client):
    """If LLM returns no 'Tags:' prefix, caption is full response, tags is empty."""
    mock_openai_client.chat.completions.create = AsyncMock(
        return_value=_mock_response("A beautiful landscape without tags.")
    )
    result = await captioner.describe(b"bytes")
    assert result.caption == "A beautiful landscape without tags."
    assert result.tags == []


async def test_describe_encodes_image_as_base64(captioner, mock_openai_client):
    mock_openai_client.chat.completions.create = AsyncMock(
        return_value=_mock_response("Caption. Tags: a, b, c")
    )
    await captioner.describe(b"image data")
    call_args = mock_openai_client.chat.completions.create.call_args
    messages = call_args.kwargs["messages"]
    content = messages[0]["content"]
    image_url = next(c for c in content if c["type"] == "image_url")
    assert image_url["image_url"]["url"].startswith("data:image/jpeg;base64,")


async def test_describe_trims_whitespace_from_caption(captioner, mock_openai_client):
    mock_openai_client.chat.completions.create = AsyncMock(
        return_value=_mock_response("  A photo.  Tags: a, b, c  ")
    )
    result = await captioner.describe(b"bytes")
    assert result.caption == "A photo."
