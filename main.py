import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Annotated, Optional

import aiohttp
import chromadb
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = os.environ.get("DATA_DIR", "/data/chroma")
VK_URL = os.environ["VK_URL"].rstrip("/")

app = FastAPI(title="Mnemonic")

chroma = chromadb.PersistentClient(path=DATA_DIR)
conversations = chroma.get_or_create_collection("conversations")
user_facts = chroma.get_or_create_collection("user_facts")


async def require_auth(
    request: Request,
    authorization: Annotated[Optional[str], Header()] = None,
    x_api_key: Annotated[Optional[str], Header()] = None,
):
    headers = {}
    if authorization:
        headers["Authorization"] = authorization
    if x_api_key:
        headers["X-API-Key"] = x_api_key
    cookie = request.headers.get("cookie")
    if cookie:
        headers["Cookie"] = cookie

    if not headers:
        raise HTTPException(status_code=401, detail="Non authentifié")

    async with aiohttp.ClientSession() as http:
        resp = await http.get(
            f"{VK_URL}/verify",
            headers={**headers, "X-Forwarded-Host": "mnemonic"},
        )
        if resp.status != 200:
            raise HTTPException(status_code=401, detail="Non authentifié")

    logger.debug(f"Requête authentifiée — {authorization or x_api_key or 'cookie'}")


Auth = Annotated[None, Depends(require_auth)]


class Message(BaseModel):
    role: str
    content: str


class SessionRequest(BaseModel):
    messages: list[Message]
    timestamp: Optional[str] = None


class Fact(BaseModel):
    type: str
    value: str


class FactsRequest(BaseModel):
    facts: list[Fact]
    session_id: str


@app.post("/users/{username}/sessions")
async def store_session(username: str, body: SessionRequest, _: Auth):
    session_id = str(uuid.uuid4())
    ts = body.timestamp or datetime.now(timezone.utc).isoformat()
    content = "\n".join(f"{m.role}: {m.content}" for m in body.messages)
    conversations.add(
        ids=[session_id],
        documents=[content],
        metadatas=[{"username": username, "timestamp": ts, "message_count": len(body.messages)}],
    )
    logger.info(f"Session {session_id} stockée pour {username} ({len(body.messages)} messages)")
    return {"session_id": session_id}


@app.post("/users/{username}/facts")
async def store_facts(username: str, body: FactsRequest, _: Auth):
    if not body.facts:
        return {"stored": 0}
    ids = [str(uuid.uuid4()) for _ in body.facts]
    ts = datetime.now(timezone.utc).isoformat()
    user_facts.add(
        ids=ids,
        documents=[f.value for f in body.facts],
        metadatas=[{
            "username": username,
            "type": f.type,
            "value": f.value,
            "session_id": body.session_id,
            "timestamp": ts,
        } for f in body.facts],
    )
    logger.info(f"{len(body.facts)} faits stockés pour {username}")
    return {"stored": len(body.facts)}


@app.get("/users/{username}/facts")
async def list_facts(username: str, _: Auth, fact_type: Optional[str] = None):
    where = {"$and": [{"username": username}, {"type": fact_type}]} if fact_type else {"username": username}
    results = user_facts.get(where=where)
    return [
        {
            "id": id_,
            "type": m["type"],
            "value": m["value"],
            "session_id": m["session_id"],
            "timestamp": m["timestamp"],
        }
        for id_, m in zip(results["ids"], results["metadatas"])
    ]


@app.get("/users/{username}/facts/search")
async def search_facts(username: str, q: str, _: Auth, n: int = 5):
    count = user_facts.count()
    if count == 0:
        return []
    results = user_facts.query(
        query_texts=[q],
        n_results=min(n, count),
        where={"username": username},
    )
    return [
        {
            "id": id_,
            "type": m["type"],
            "value": m["value"],
            "session_id": m["session_id"],
        }
        for id_, m in zip(results["ids"][0], results["metadatas"][0])
    ]


@app.get("/users/{username}/sessions/{session_id}")
async def get_session(username: str, session_id: str, _: Auth):
    results = conversations.get(ids=[session_id])
    if not results["ids"]:
        raise HTTPException(status_code=404, detail="Session introuvable")
    meta = results["metadatas"][0]
    if meta["username"] != username:
        raise HTTPException(status_code=403, detail="Accès refusé")
    return {"session_id": session_id, "content": results["documents"][0], **meta}
