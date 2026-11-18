__all__ = ["SlackWebhookNotifier", "SlackBotNotifier"]

# Standard library imports
import json
import logging
from pathlib import Path
import requests
import os

# Related third party imports
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from slack_sdk.web.slack_response import SlackResponse
from tenacity import (
    retry,
    retry_if_exception_type,
    retry_if_result,
    stop_after_attempt,
    wait_exponential,
    RetryCallState,
)

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

_R_MAX_STOPS = 10
_R_MAX_SECS = 15


def is_false(value: bool):
    return value is False


def log_final_error(retry_state: RetryCallState):
    exception = retry_state.outcome.exception()
    if exception:
        logger.error(
            f"Failed to send Slack notification after multiple attempts: {exception}"
        )


class SlackWebhookNotifier:
    """Notifies a Slack channel through a webhook (one-way communication via POST)."""

    def __init__(
        self,
        webhook_url=os.getenv("SLACK_WEBHOOK_URL"),
        channel="#spherex-process",
    ):
        self.webhook_url = webhook_url
        self.channel = channel

        if not self.webhook_url:
            raise ValueError("SLACK_WEBHOOK_URL environment variable is not set")

    @staticmethod
    def _format_message(message: str, title: str) -> dict:
        """Format a message for the Slack bot to notify a channel about.

        Parameters
        ----------
        message : `str`
            Message bot will present.
        title : `str`
            Title of message.

        Returns
        -------
        `dict`
            Payload that includes message written to the Slack channel by the
            Slack bot.

        """
        return {
            "attachments": [
                {
                    "color": "#9733EE",
                    "blocks": [
                        {
                            "type": "header",
                            "text": {
                                "type": "plain_text",
                                "text": title,
                            },
                        },
                        {"type": "divider"},
                        {
                            "type": "section",
                            "text": {
                                "type": "mrkdwn",
                                "text": message,
                            },
                        },
                    ],
                }
            ]
        }

    @retry(
        stop=stop_after_attempt(_R_MAX_STOPS),
        wait=wait_exponential(multiplier=1, min=4, max=_R_MAX_SECS),
        retry=(
            retry_if_result(is_false)
            | retry_if_exception_type(
                (
                    requests.exceptions.ConnectTimeout,
                    requests.exceptions.ReadTimeout,
                    requests.exceptions.HTTPError,
                    requests.exceptions.RequestException,
                )
            )
        ),
        retry_error_callback=log_final_error,
    )
    def send_notification(
        self, message: str, title: str = "Processing Summary"
    ) -> None:
        """Send the notification to channel through Slack bot.

        Parameters
        ----------
        message : `str`
            Message bot will present.
        title : `str`
            Title of message, by default "Processing Summary".

        """
        payload = self._format_message(message, title)

        response = requests.post(
            self.webhook_url,
            data=json.dumps(payload),
            headers={"Content-Type": "application/json"},
            timeout=10,
        )
        response.raise_for_status()

        logger.info(f"Slack channel {self.channel} messaged successfully.")


class SlackBotNotifier:
    """Notifies a Slack channel through a token (one-way communication via POST)."""

    def __init__(
        self,
        bot_token=os.getenv("SLACK_BOT_TOKEN"),
        channel_id=os.getenv("SLACK_CHANNEL_ID"),
        channel="#spherex-process",
    ):
        self.bot_token = bot_token
        self.channel_id = channel_id
        self.channel = channel

        required_vars = {
            "SLACK_BOT_TOKEN": self.bot_token,
            "SLACK_CHANNEL_ID": self.channel_id,
        }

        missing_vars = [name for name, value in required_vars.items() if not value]
        if missing_vars:
            raise ValueError(
                f"Missing required environment variables: {', '.join(missing_vars)}"
            )

    def _get_file_upload_url(
        self, file: Path, slack_client: WebClient
    ) -> SlackResponse:
        """Get the Slack upload URL for file.

        Parameters
        ----------
        file : `Path`
            Path to file to upload.
        slack_client : `WebClient`
            Slack web client.

        Returns
        -------
        `SlackResponse`
            Slack URL to upload files to.

        """
        upload_url = slack_client.files_getUploadURLExternal(
            token=self.bot_token,
            filename=file.name,
            length=file.stat().st_size,
        )

        if not upload_url["ok"]:
            raise ValueError(
                f"Unable to retrieve the URL for uploading the attachment to Slack: \n{upload_url}"
            )

        return upload_url

    @retry(
        stop=stop_after_attempt(_R_MAX_STOPS),
        wait=wait_exponential(multiplier=1, min=4, max=_R_MAX_SECS),
        retry=(
            retry_if_result(is_false)
            | retry_if_exception_type(
                (
                    requests.exceptions.ConnectTimeout,
                    requests.exceptions.ReadTimeout,
                    requests.exceptions.HTTPError,
                    requests.exceptions.RequestException,
                )
            )
        ),
        retry_error_callback=log_final_error,
    )
    def _upload_file_to_url(self, file: Path, upload_url: SlackResponse) -> None:
        """Upload file to specified Slack URL.

        Parameters
        ----------
        file : `Path`
            Path to file to upload.
        upload_url : `SlackResponse`
            Upload url.

        """
        payload = {
            "filename": file.name,
            "token": self.bot_token,
        }

        with file.open("rb") as f:
            files = {
                "file": f,
            }

            response = requests.post(
                upload_url["upload_url"], files=files, params=payload
            )

            response.raise_for_status()
            logger.info(f"Slack channel {self.channel} messaged successfully.")

    def _make_file_accessible(
        self,
        file: Path,
        slack_client: WebClient,
        upload_url: SlackResponse,
        message: str,
    ) -> None:
        """Once the file is uploaded to Slack, this function will ensure
        that it is accessible to all members of the channel.

        Parameters
        ----------
        file : `Path`
            Path to file to upload.
        slack_client : `WebClient`
            Slack web client.
        upload_url : `SlackResponse`
            Upload url.
        message : `str`
            Slack message to send.

        """
        slack_client.files_completeUploadExternal(
            token=self.bot_token,
            files=[{"id": upload_url["file_id"], "title": file.name}],
            channel_id=self.channel_id,
            initial_comment=message,
        )

    def upload_file(self, file: Path, message: str) -> None:
        """Upload file to Slack channel.

        Parameters
        ----------
        file : `Path`
            Path to file to upload.
        message : `str`
            Slack message to send.

        """
        try:
            slack_client = WebClient(token=self.bot_token)

            upload_url = self._get_file_upload_url(file, slack_client)
            self._upload_file_to_url(file, upload_url)
            self._make_file_accessible(file, slack_client, upload_url, message)
        except SlackApiError as e:
            logging.error(
                f"Slack API Error occurred when uploading file to to {self.channel}: {e.response['error']}"
            )
        except Exception as e:
            logging.error(
                f"An error occurred when uploading file to {self.channel}: {e}"
            )
