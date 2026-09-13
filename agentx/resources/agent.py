from typing import Optional, List
from pydantic import BaseModel, PrivateAttr, Field
import requests
import os
import logging
from agentx.util import get_headers, api_base
from .conversation import Conversation


class Agent(BaseModel):
    id: str = Field(alias="_id")
    name: str
    avatar: Optional[str] = None
    createdAt: Optional[str] = None
    updatedAt: Optional[str] = None


    # Hosted credentials threaded from the constructing AgentX client. The module-level
    # api_base()/get_headers() read only env vars, and the client deliberately stopped
    # writing its constructor args into os.environ - without binding, a client created with
    # api_key=/base_url= issued these calls unauthenticated against the default host.
    _api_key: Optional[str] = PrivateAttr(default=None)
    _base_url: Optional[str] = PrivateAttr(default=None)

    def _bind(self, api_key: Optional[str], base_url: Optional[str]) -> "Agent":
        self._api_key = api_key
        self._base_url = base_url
        return self

    def _api_base(self) -> str:
        return self._base_url or api_base()

    def _headers(self):
        return get_headers(self._api_key)

    def __init__(self, **data):
        super().__init__(**data)

    def new_conversation(self) -> Conversation:
        url = f"{self._api_base()}/access/agents/{self.id}/conversations/new"
        response = requests.post(url, headers=self._headers(), json={"type": "chat"})
        if response.status_code == 200:
            data = response.json()
            data["agent_id"] = self.id
            return Conversation(**data)._bind(self._api_key, self._base_url)
        else:
            raise Exception(f"Failed to create conversation: {response.reason}")

    def get_conversation(self, id: str) -> Conversation:
        list_of_conversations = self.list_conversations()
        return next(
            (conv for conv in list_of_conversations if conv.id == id),
            Exception("404 - Conversation not found"),
        )

    def list_conversations(self) -> List[Conversation]:
        url = f"{self._api_base()}/access/agents/{self.id}/conversations"
        response = requests.get(url, headers=self._headers())
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
                    # Bound to this agent's own credentials (threaded in via Agent._bind) so
                    # the conversation's calls authenticate the same way this listing did.
                )._bind(self._api_key, self._base_url)
                for conv_res in response.json()
            ]
        else:
            raise Exception(f"Failed to retrieve agent details: {response.reason}")
