from typing import Optional, List
from pydantic import BaseModel, Field
import requests
import os
import logging
from dataclasses import dataclass
from agentx.util import get_headers, api_base
from .conversation import Conversation


@dataclass
class Agent(BaseModel):
    id: str = Field(alias="_id")
    name: str
    avatar: Optional[str]
    createdAt: Optional[str]
    updatedAt: Optional[str]

    def __init__(self, **data):
        super().__init__(**data)

    def get_conversation(self, id: str) -> Conversation:
        list_of_conversations = self.list_conversations()
        return next(
            (conv for conv in list_of_conversations if conv.id == id),
            Exception("404 - Conversation not found"),
        )

    def list_conversations(self) -> List[Conversation]:
        url = f"{api_base()}/access/agents/{self.id}/conversations"
        response = requests.get(url, headers=get_headers())
        if response.status_code == 200:
            return [
                Conversation(
                    agent_id=self.id,
                    id=conv_res.get("_id"),
                    title=conv_res.get("title"),
                    users=conv_res.get("users"),
                    agents=conv_res.get("bots"),
                    createdAt=conv_res.get("createdAt"),
                    updatedAt=conv_res.get("updatedAt"),
                )
                for conv_res in response.json()
            ]
        else:
            raise Exception(f"Failed to retrieve agent details: {response.reason}")
