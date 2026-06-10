from typing import Optional, List, Dict, Any, Iterator
from pydantic import BaseModel, Field
import requests
import os
import json
import logging
from agentx.util import get_headers, api_base
from agentx.resources.agent import Agent
from agentx.resources.conversation import Conversation, ChatResponse


class User(BaseModel):
    id: str = Field(alias="_id")
    name: str
    email: str
    deleted: bool
    createdAt: str
    updatedAt: str
    avatar: str
    status: int
    customer: str
    resetPwdToken: Optional[str] = None
    defaultWorkspace: Optional[str] = None
    workspaces: List[str] = []

    class Config:
        populate_by_name = True
        extra = "ignore"


class Workforce(BaseModel):
    id: str = Field(alias="_id")
    agents: List[Agent]
    name: str
    image: str
    description: str
    manager: Agent
    creator: User
    context: int
    references: bool
    workspace: Optional[str] = None
    createdAt: str
    updatedAt: str

    class Config:
        populate_by_name = True
        extra = "ignore"

    def new_conversation(self) -> Conversation:
        """Create a new conversation for this workforce."""
        url = f"{api_base()}/access/teams/{self.id}/conversations/new"
        response = requests.post(
            url,
            headers=get_headers(),
            json={"type": "chat"},
        )
        if response.status_code == 200:
            conv_data = response.json()
            # Set the agent_id to the manager's ID since this is a workforce conversation
            conv_data["agent_id"] = self.manager.id
            return Conversation(**conv_data)
        else:
            raise Exception(
                f"Failed to create new conversation: {response.status_code} - {response.reason}"
            )

    def list_conversations(self) -> List[Conversation]:
        """List all conversations for this workforce."""
        url = f"{api_base()}/access/teams/{self.id}/conversations"
        response = requests.get(url, headers=get_headers())
        if response.status_code == 200:
            conversations = []
            for conv_data in response.json():
                # Set the agent_id to the manager's ID since this is a workforce conversation
                conv_data["agent_id"] = self.manager.id
                conversations.append(Conversation(**conv_data))
            return conversations
        else:
            raise Exception(
                f"Failed to list conversations: {response.status_code} - {response.reason}"
            )

    def chat_stream(
        self, conversation_id: str, message: str, context: int = -1
    ) -> Iterator[ChatResponse]:
        """Send a message to a team conversation and stream the response."""
        url = (
            f"{api_base()}/access/teams/conversations/{conversation_id}/jsonmessagesse"
        )
        response = requests.post(
            url, headers=get_headers(), json={"message": message, "context": context}
        )
        result = ""
        if response.status_code == 200:
            buf = b""
            for chunk in response.iter_content():
                buf += chunk
                try:
                    chunk = buf.decode("utf-8")
                except UnicodeDecodeError:
                    continue
                result += chunk
                buf = b""
                try:
                    if result.count("{") == result.count("}"):
                        catch_json = json.loads(result)
                        if catch_json:
                            result = ""
                            yield ChatResponse(
                                text=catch_json.get("text"),
                                cot=catch_json.get("cot"),
                                botId=catch_json.get("botId"),
                                reference=catch_json.get("reference"),
                                tasks=catch_json.get("tasks"),
                            )
                except json.JSONDecodeError:
                    continue
        else:
            raise Exception(
                f"Failed to send message: {response.status_code} - {response.reason}"
            )
