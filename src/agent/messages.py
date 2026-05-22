"""Conversation state for the LLM agent.

MessageManager owns the system message and the alternating user/assistant turns.
It produces provider-specific kwargs via to_X_kwargs methods.
"""

from typing import Literal

from pydantic import BaseModel, Field


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str | list[dict]


class MessageManager(BaseModel):
    system: str = ""
    messages: list[Message] = Field(default_factory=list)

    def add_user_message(self, content: str | list[dict]) -> None:
        self.messages.append(Message(role="user", content=content))

    def add_assistant_message(self, content: str | list[dict]) -> None:
        self.messages.append(Message(role="assistant", content=content))

    def to_anthropic_kwargs(self) -> dict:
        """Returns kwargs to splat into Anthropic Messages API calls."""
        return {
            "system": self.system,
            "messages": [m.model_dump() for m in self.messages],
        }
