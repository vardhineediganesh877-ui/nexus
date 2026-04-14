"""
NEXUS Telegram Bot Runner — Real Telegram Bot API integration.

Uses long-polling to receive commands and send responses.
Works with python-telegram-bot library or raw HTTP API.

Two backends:
1. TelegramAPIPoller — lightweight, uses aiohttp directly (no deps)
2. python_telegram_bot — full-featured, uses python-telegram-bot library
"""

import asyncio
import json
import logging
import os
import time
from typing import Optional, Callable

import aiohttp

from ..config import NexusConfig
from ..models import TradeSignal
from .bot import TelegramBot
from .alerts import AlertManager, AlertRule, Alert

logger = logging.getLogger(__name__)


class TelegramAPIPoller:
    """
    Lightweight Telegram Bot API client using long-polling.
    Zero external dependencies beyond aiohttp.
    """

    API_BASE = "https://api.telegram.org/bot{token}/{method}"

    def __init__(self, token: str, chat_id: str, nexus_bot: TelegramBot,
                 alert_manager: Optional[AlertManager] = None):
        self.token = token
        self.chat_id = chat_id
        self.bot = nexus_bot
        self.alert_manager = alert_manager or AlertManager(
            send_fn=self.send_message
        )
        self._last_update_id = 0
        self._running = False
        self._session: Optional[aiohttp.ClientSession] = None
        self._poll_timeout = 30
        self._allowed_users: set = set()  # Empty = allow all

    async def start(self):
        """Start the bot polling loop."""
        if not self.token:
            raise ValueError("TELEGRAM_BOT_TOKEN not configured")

        self._session = aiohttp.ClientSession()
        self._running = True

        # Verify bot token
        me = await self._api_call("getMe")
        if not me or not me.get("ok"):
            raise ValueError(f"Invalid bot token: {me}")

        bot_name = me["result"]["username"]
        logger.info(f"Telegram bot started: @{bot_name}")

        # Set commands
        await self._set_commands()

        # Send startup message
        await self.send_message("⚡ *NEXUS Online* — AI Trading Intelligence ready.\nType /help for commands.")

        # Start polling
        await self._poll_loop()

    async def stop(self):
        """Stop the bot."""
        self._running = False
        if self._session:
            await self._session.close()
            self._session = None
        logger.info("Telegram bot stopped")

    async def send_message(self, text: str, chat_id: Optional[str] = None,
                           parse_mode: str = "Markdown") -> dict:
        """Send a message to Telegram."""
        if not self._session:
            # Create session if needed (for alert_manager calls before start)
            self._session = aiohttp.ClientSession()

        payload = {
            "chat_id": chat_id or self.chat_id,
            "text": text,
            "parse_mode": parse_mode,
        }

        try:
            result = await self._api_call("sendMessage", payload)
            return result
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            return {}

    async def _poll_loop(self):
        """Main polling loop."""
        while self._running:
            try:
                updates = await self._get_updates()

                for update in updates:
                    self._last_update_id = update["update_id"] + 1
                    await self._handle_update(update)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Poll error: {e}")
                await asyncio.sleep(5)

    async def _get_updates(self) -> list:
        """Fetch updates via long-polling."""
        payload = {
            "offset": self._last_update_id,
            "timeout": self._poll_timeout,
            "allowed_updates": ["message"],
        }
        result = await self._api_call("getUpdates", payload)
        if result and result.get("ok"):
            return result.get("result", [])
        return []

    async def _handle_update(self, update: dict):
        """Process an incoming update."""
        message = update.get("message")
        if not message:
            return

        chat_id = str(message["chat"]["id"])
        text = message.get("text", "")
        user = message.get("from", {})

        # Auth check
        if self._allowed_users and user.get("id") not in self._allowed_users:
            await self.send_message("⛔ Unauthorized.", chat_id)
            return

        if not text.startswith("/"):
            return

        logger.info(f"TG command from {user.get('username', 'unknown')}: {text}")

        # Process via TelegramBot
        response = self.bot.process_message(text)

        # Send response
        try:
            await self.send_message(response, chat_id)
        except Exception as e:
            # Try without markdown if parsing fails
            logger.warning(f"Markdown parse failed, sending plain: {e}")
            await self.send_message(response, chat_id, parse_mode="")

    async def _set_commands(self):
        """Set bot commands in Telegram UI."""
        commands = [
            {"command": cmd, "description": desc}
            for cmd, desc in TelegramBot.COMMANDS.items()
        ]
        await self._api_call("setMyCommands", {"commands": commands})

    async def _api_call(self, method: str, payload: Optional[dict] = None) -> Optional[dict]:
        """Make a Telegram Bot API call."""
        url = self.API_BASE.format(token=self.token, method=method)

        try:
            async with self._session.post(url, json=payload or {},
                                          timeout=aiohttp.ClientTimeout(total=35)) as resp:
                return await resp.json()
        except asyncio.TimeoutError:
            # Normal for long-polling
            return None
        except Exception as e:
            logger.error(f"API call failed ({method}): {e}")
            return None


def create_bot(config: NexusConfig) -> Optional[TelegramAPIPoller]:
    """Factory: create a configured Telegram poller from NexusConfig."""
    if not config.telegram_bot_token:
        logger.warning("No TELEGRAM_BOT_TOKEN — Telegram alerts disabled")
        return None

    nexus_bot = TelegramBot(config)
    poller = TelegramAPIPoller(
        token=config.telegram_bot_token,
        chat_id=config.telegram_chat_id,
        nexus_bot=nexus_bot,
    )
    return poller
