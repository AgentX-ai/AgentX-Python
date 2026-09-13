import json
import requests
from typing import Optional, List, Any, Iterator
from pydantic import BaseModel, PrivateAttr, Field
from agentx.util import get_headers, api_base


class ChatResponse(BaseModel):
    text: Optional[str]
    cot: Optional[str]
    botId: str
    reference: Optional[Any]
    tasks: Optional[Any]


class Message(BaseModel):
    id: str = Field(alias="_id")
    conversationId: str
    role: str  # user or bot
    botId: Optional[str] = Field(alias="bot", default=None)
    userId: Optional[str] = Field(alias="user", default=None)
    text: Optional[str]
    cot: Optional[str]
    createdAt: str
    updatedAt: str

    class Config:
        populate_by_name = True
        extra = "ignore"


class Conversation(BaseModel):
    agent_id: str
    id: str = Field(alias="_id")
    title: Optional[str] = Field(default=None)  # conversation customized title
    users: List[str]
    agents: List[str] = Field(alias="bots")
    createdAt: Optional[str]
    updatedAt: Optional[str]

    class Config:
        populate_by_name = True
        extra = "ignore"


    # Hosted credentials threaded from the constructing AgentX client. The module-level
    # api_base()/get_headers() read only env vars, and the client deliberately stopped
    # writing its constructor args into os.environ - without binding, a client created with
    # api_key=/base_url= issued these calls unauthenticated against the default host.
    _api_key: Optional[str] = PrivateAttr(default=None)
    _base_url: Optional[str] = PrivateAttr(default=None)

    def _bind(self, api_key: Optional[str], base_url: Optional[str]) -> "Conversation":
        self._api_key = api_key
        self._base_url = base_url
        return self

    def _api_base(self) -> str:
        return self._base_url or api_base()

    def _headers(self):
        return get_headers(self._api_key)

    def __init__(self, **data):
        super().__init__(**data)

    def new_conversation(self) -> "Conversation":
        url = f"{self._api_base()}/access/agents/{self.agent_id}/conversations/new"
        response = requests.post(
            url,
            headers=self._headers(),
            json={"type": "chat"},
        )
        if response.status_code == 200:
            new_conv = response.json()
            new_conv["agent_id"] = self.agent_id
            # Bound to this conversation's own credentials - unbound, the sibling
            # conversation fell back to env credentials against the default host.
            return Conversation(**new_conv)._bind(self._api_key, self._base_url)
        else:
            raise Exception(
                f"Failed to create new conversation: {response.status_code} - {response.reason}"
            )

    def list_messages(self) -> List[Message]:
        url = f"{self._api_base()}/access/agents/{self.agent_id}/conversations/{self.id}"
        response = requests.get(url, headers=self._headers())
        if response.status_code == 200:
            res = response.json()
            if res.get("messages"):
                message_list = []
                for message in res["messages"]:
                    if message.get("bot"):
                        message["role"] = "bot"
                    else:
                        message["role"] = "user"
                    message_list.append(Message(**message))
                return message_list
            else:
                raise Exception("No messages found in the conversation.")
        else:
            raise Exception(
                f"Failed to retrieve agent details: {response.status_code} - {response.reason}"
            )

    def chat(self, message: str, context: Optional[int] = None):
        url = f"{self._api_base()}/access/conversations/{self.id}/message"
        response = requests.post(
            url,
            headers=self._headers(),
            json={"message": message, "context": context},
        )
        return response.json()

    def chat_stream(self, message: str, context: Optional[int] = None) -> Iterator[ChatResponse]:
        url = f"{self._api_base()}/access/conversations/{self.id}/jsonmessagesse"
        response = requests.post(
            url, headers=self._headers(), json={"message": message, "context": context}
        )
        result = ""
        if response.status_code == 200:
            buf = b""
            for chunk in response.iter_content():
                buf += chunk
                try:
                    chunk = buf.decode("utf-8")
                    # yield chunk
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
