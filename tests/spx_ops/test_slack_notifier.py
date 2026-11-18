import json
from unittest.mock import Mock, patch
import pytest
from spx_ops.slack_notifier import SlackWebhookNotifier, SlackBotNotifier


def test_webhook_notifier_requires_webhook_url():
    with pytest.raises(ValueError, match="SLACK_WEBHOOK_URL"):
        SlackWebhookNotifier(webhook_url=None)


def test_webhook_format_message():
    payload = SlackWebhookNotifier._format_message(
        message="Pipeline finished",
        title="Summary",
    )

    attachment = payload["attachments"][0]
    blocks = attachment["blocks"]

    assert attachment["color"] == "#9733EE"
    assert blocks[0]["type"] == "header"
    assert blocks[0]["text"]["text"] == "Summary"
    assert blocks[2]["text"]["text"] == "Pipeline finished"


@patch("spx_ops.slack_notifier.requests.post")
def test_webhook_send_notification_posts_payload(mock_post):
    response = Mock()
    response.raise_for_status = Mock()
    mock_post.return_value = response

    notifier = SlackWebhookNotifier(
        webhook_url="https://example.com/webhook",
        channel="#test",
    )

    notifier.send_notification("Done", title="Run complete")

    mock_post.assert_called_once()
    _, kwargs = mock_post.call_args

    assert kwargs["headers"] == {"Content-Type": "application/json"}
    assert kwargs["timeout"] == 10

    payload = json.loads(kwargs["data"])
    assert payload["attachments"][0]["blocks"][0]["text"]["text"] == "Run complete"
    assert payload["attachments"][0]["blocks"][2]["text"]["text"] == "Done"

    response.raise_for_status.assert_called_once()


def test_bot_notifier_requires_bot_token_and_channel_id():
    with pytest.raises(ValueError, match="SLACK_BOT_TOKEN, SLACK_CHANNEL_ID"):
        SlackBotNotifier(bot_token=None, channel_id=None)

    with pytest.raises(ValueError, match="SLACK_CHANNEL_ID"):
        SlackBotNotifier(bot_token="token", channel_id=None)

    with pytest.raises(ValueError, match="SLACK_BOT_TOKEN"):
        SlackBotNotifier(bot_token=None, channel_id="channel")


def test_get_file_upload_url(tmp_path):
    file = tmp_path / "summary.txt"
    file.write_text("hello")

    slack_client = Mock()
    slack_client.files_getUploadURLExternal.return_value = {
        "ok": True,
        "upload_url": "https://upload.example.com",
        "file_id": "F123",
    }

    notifier = SlackBotNotifier(bot_token="token", channel_id="C123")

    result = notifier._get_file_upload_url(file, slack_client)

    slack_client.files_getUploadURLExternal.assert_called_once_with(
        token="token",
        filename="summary.txt",
        length=file.stat().st_size,
    )
    assert result["upload_url"] == "https://upload.example.com"
    assert result["file_id"] == "F123"


def test_get_file_upload_url_raises_if_not_ok(tmp_path):
    file = tmp_path / "summary.txt"
    file.write_text("hello")

    slack_client = Mock()
    slack_client.files_getUploadURLExternal.return_value = {
        "ok": False,
        "error": "bad_request",
    }

    notifier = SlackBotNotifier(bot_token="token", channel_id="C123")

    with pytest.raises(ValueError, match="Unable to retrieve the URL"):
        notifier._get_file_upload_url(file, slack_client)


@patch("spx_ops.slack_notifier.requests.post")
def test_upload_file_to_url_posts_file(mock_post, tmp_path):
    file = tmp_path / "summary.txt"
    file.write_text("hello")

    response = Mock()
    response.raise_for_status = Mock()
    mock_post.return_value = response

    notifier = SlackBotNotifier(bot_token="token", channel_id="C123")

    notifier._upload_file_to_url(
        file,
        {"upload_url": "https://upload.example.com"},
    )

    mock_post.assert_called_once()
    _, kwargs = mock_post.call_args

    assert kwargs["params"] == {
        "filename": "summary.txt",
        "token": "token",
    }
    assert "file" in kwargs["files"]
    response.raise_for_status.assert_called_once()


def test_make_file_accessible(tmp_path):
    file = tmp_path / "summary.txt"
    file.write_text("hello")

    slack_client = Mock()
    notifier = SlackBotNotifier(bot_token="token", channel_id="C123")

    notifier._make_file_accessible(
        file=file,
        slack_client=slack_client,
        upload_url={"file_id": "F123"},
        message="Here is the file",
    )

    slack_client.files_completeUploadExternal.assert_called_once_with(
        token="token",
        files=[{"id": "F123", "title": "summary.txt"}],
        channel_id="C123",
        initial_comment="Here is the file",
    )


@patch("spx_ops.slack_notifier.WebClient")
def test_upload_file_full_flow(mock_webclient, tmp_path):
    file = tmp_path / "summary.txt"
    file.write_text("hello")

    slack_client = Mock()
    mock_webclient.return_value = slack_client

    notifier = SlackBotNotifier(bot_token="token", channel_id="C123")

    with patch.object(
        notifier,
        "_get_file_upload_url",
        return_value={"upload_url": "https://upload.example.com", "file_id": "F123"},
    ) as mock_get_url, patch.object(
        notifier,
        "_upload_file_to_url",
    ) as mock_upload, patch.object(
        notifier,
        "_make_file_accessible",
    ) as mock_accessible:

        notifier.upload_file(file, message="Run summary attached")

    mock_webclient.assert_called_once_with(token="token")
    mock_get_url.assert_called_once_with(file, slack_client)
    mock_upload.assert_called_once_with(
        file,
        {"upload_url": "https://upload.example.com", "file_id": "F123"},
    )
    mock_accessible.assert_called_once_with(
        file,
        slack_client,
        {"upload_url": "https://upload.example.com", "file_id": "F123"},
        "Run summary attached",
    )
