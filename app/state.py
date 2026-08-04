"""Graph state shared across nodes for a single chat session (thread)."""
from __future__ import annotations

from typing import Annotated, Optional, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class DogProfile(BaseModel):
    """Structured facts Polo has learned about the customer's dog so far."""

    name: Optional[str] = Field(default=None, description="The dog's name")
    breed: Optional[str] = Field(default=None, description="The dog's breed")
    age: Optional[str] = Field(default=None, description="The dog's age, e.g. '2 years' or 'puppy'")
    size: Optional[str] = Field(
        default=None, description="Size category if known: small, medium, or large"
    )
    notes: Optional[str] = Field(
        default=None,
        description="Any other relevant detail: allergies, coat type, activity level, etc.",
    )

    def is_empty(self) -> bool:
        return not any([self.name, self.breed, self.age, self.size, self.notes])

    def summary(self) -> str:
        if self.is_empty():
            return "No details collected yet."
        parts = []
        if self.name:
            parts.append(f"name={self.name}")
        if self.breed:
            parts.append(f"breed={self.breed}")
        if self.age:
            parts.append(f"age={self.age}")
        if self.size:
            parts.append(f"size={self.size}")
        if self.notes:
            parts.append(f"notes={self.notes}")
        return ", ".join(parts)


class ChatState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    dog_profile: dict  # serialized DogProfile
