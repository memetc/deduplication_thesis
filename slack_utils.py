import os
import logging
from slack_sdk import WebClient
from dotenv import load_dotenv

def send_slack_message(message, channel="reports", username="Reporter", token=None):
    """
    Sends a message to a Slack channel using slack_sdk's WebClient.

    Parameters
    ----------
    message : str
        The message text to send.
    channel : str, optional
        The Slack channel where the message will be posted (default is "bot-updates").
    username : str, optional
        The username to post the message as (default is "Bot User").
    token : str, optional
        The Slack OAuth token. If not provided, the function will attempt to use the 
        SLACK_TOKEN environment variable.

    Returns
    -------
    response : dict
        The response from the Slack API if the message is successfully sent.

    Raises
    ------
    ValueError
        If the Slack token is not provided and not found in the environment variables.
    Exception
        Any exception raised by the Slack API call.
    """
    
    load_dotenv()
    # Use token from argument or environment variable
    if token is None:
        token = os.getenv("SLACK_TOKEN")
        if token is None:
            raise ValueError("Slack token must be provided or set in the SLACK_TOKEN environment variable.")
    
    client = WebClient(token=token)
    
    try:
        response = client.chat_postMessage(
            channel=channel,
            text=message,
            username=username
        )
        logging.info("Message sent successfully to Slack channel '%s'.", channel)
        return response
    except Exception as e:
        logging.error("Failed to send Slack message: %s", e)
        raise

