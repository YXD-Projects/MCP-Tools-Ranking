# ================================================================
# Filter Service (FAISS dual-index, pass-through output)
#
# Flow:
#   1) /index_tools: take the EXACT MCP tools/list payload, embed & store.
#   2) /filter-tools: given session_id + query, return SAME structure
#      with Top-K tools selected using two-stage retrieval
#      (actions → entities re-rank) and dynamic-K.
#
# 
# ================================================================

from __future__ import annotations

import os, re, time, logging
from contextlib import asynccontextmanager
from typing import List, Any, Tuple, Dict, Optional
from dataclasses import dataclass, field

import faiss
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Use MCP canonical types so our I/O matches tools/list exactly.
from mcp.types import Tool as McpTool, ListToolsResult

# MiniLM encoder 
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("filter_svc")


# --------------------------
# knobs (weights, dims, FAISS params)
# --------------------------
def _env_float(k: str, d: float) -> float:
    v = os.getenv(k)
    try:
        return float(v) if v is not None else d
    except:
        return d

def _env_int(k: str, d: int) -> int:
    v = os.getenv(k)
    try:
        return int(v) if v is not None else d
    except:
        return d

A_W = _env_float("FILTER_A_W", 0.6)                 # action weight
E_W = _env_float("FILTER_E_W", 0.4)                 # entity weight
NLIST_HINT = _env_int("FILTER_NLIST_HINT", 0)       # 0 -> heuristic

# Dynamic-K knobs
MIN_SCORE   = _env_float("FILTER_MIN_SCORE", 0.25)  # minimum fused score to be "good enough"
MIN_K       = _env_int("FILTER_MIN_K", 5)           # always return at least this many (if available)
MAX_K       = _env_int("FILTER_MAX_K", 25)          # never return more than this many
GAP_DROP    = _env_float("FILTER_GAP_DROP", 0.20)   # stop at first big score drop ("elbow")

# --------------------------
# Embeddings: MiniLM encoder
# --------------------------
_EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "all-MiniLM-L6-v2")
_encoder: Optional[SentenceTransformer] = None

def get_encoder() -> SentenceTransformer:
    global _encoder
    if _encoder is None:
        # TRANSFORMERS_NO_TF=1 in the shell to avoid tf_keras import
        logger.info(f"[FILTER] loading encoder: {_EMBED_MODEL_NAME}")
        _encoder = SentenceTransformer(_EMBED_MODEL_NAME)
    return _encoder


def embed_texts(texts: List[str]) -> np.ndarray:
    enc = get_encoder()
    # We’ll not normalize here; we do it with faiss to keep behavior explicit.
    vecs = enc.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=False,
        batch_size=int(os.getenv("EMBED_BATCH", "64")),
    )

    # Ensure float32 for faiss
    if vecs.dtype != np.float32:
        vecs = vecs.astype(np.float32, copy=False)
    return vecs

# Utility: L2-normalize vectors for cosine/IP search equivalence.
def l2_normalize(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    faiss.normalize_L2(x)
    return x


# --------------------------
# Lightweight NLP helpers
# --------------------------
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_\-]+")
VERB_HINTS = {"create","list","search","update","delete","add","remove","open"}

def dedup(seq: List[str]) -> List[str]:
    return list(dict.fromkeys(seq))

def tokenize(text: str) -> List[str]:
    return [w.lower() for w in _WORD_RE.findall(text or "")]

# extract likely action tokens (verbs/phrases) from description.
def extract_actions_from_text(text: str) -> List[str]:
    toks = tokenize(text)
    verbs = [w for w in toks if w in VERB_HINTS or w.endswith(("e","ed","ing","t","n"))]
    verbs = dedup(verbs)
    return verbs or toks[:5]

# extract likely entity names from a JSON schema.
def extract_entities_from_schema(schema: Dict[str, Any], name: str = "", desc: str = "") -> List[str]:
    
    ents: List[str] = []

    # --- 1. From schema ---
    if isinstance(schema, dict):
        props = schema.get("properties") or {}
        if isinstance(props, dict):
            ents.extend(map(str, props.keys()))
            for v in props.values():
                if isinstance(v, dict) and isinstance(v.get("enum"), list):
                    ents.extend(map(str, v["enum"]))
        req = schema.get("required")
        if isinstance(req, list):
            ents.extend(map(str, req))

    # --- 2. From name/description ---
    text = f"{name or ''} {desc or ''}".lower()
    words = re.findall(r"[a-z][a-z0-9_\-]+", text)
    # noun-ish tokens: avoid verbs, short junk, or filler words
    for w in words:
        if len(w) > 3 and not w.endswith(("ed", "ing")):
            ents.append(w)

    # --- 3. Normalize + dedup ---
    ents = [w.strip().lower() for w in ents if w.strip()]
    return dedup(ents)


# Produce (actions, entities) features for a single tool.
def extract_actions_entities_for_tool(name: str, desc: str, schema: Dict[str, Any]) -> Tuple[str, str]:
    actions = extract_actions_from_text(f"{name} {desc}")
    entities = extract_entities_from_schema(schema, name, desc)
    entities += [w for w in tokenize(desc) if w.endswith(("tion","ment","ness","ship","ity","er","or"))]
    return " ".join(actions) or (name or ""), " ".join(dedup(entities)) or (desc or name or "")


# --------------------------
# FAISS helpers
# --------------------------
def _choose_nlist(n: int) -> int:
    if n <= 64:
        return 0
    return int(max(2, min(1024, round(n ** 0.5))))

def _build_index(xb: np.ndarray, label: str) -> tuple[faiss.Index, int]:
    n, d = xb.shape
    nlist = NLIST_HINT or _choose_nlist(n)
    if nlist < 2:
        base = faiss.index_factory(d, "Flat", faiss.METRIC_INNER_PRODUCT)
        idx = faiss.IndexIDMap2(base)
        idx.add_with_ids(xb, np.arange(n, dtype=np.int64))
        logger.info(f"[FILTER] {label}: FlatIP n={n} d={d}")
        return idx, 0
    try:
        ivf = faiss.index_factory(d, f"IVF{nlist},Flat", faiss.METRIC_INNER_PRODUCT)
        while hasattr(ivf, "nlist") and n < 39 * ivf.nlist and ivf.nlist > 1:
            ivf = faiss.index_factory(d, f"IVF{max(1, ivf.nlist//2)},Flat", faiss.METRIC_INNER_PRODUCT)
        if hasattr(ivf, "train"):
            ivf.train(xb)
        idx = faiss.IndexIDMap2(ivf)
        idx.add_with_ids(xb, np.arange(n, dtype=np.int64))
        nl = getattr(ivf, "nlist", 0)
        logger.info(f"[FILTER] {label}: IVF-Flat(IP) nlist={nl} n={n} d={d}")
        return idx, nl
    except Exception as e:
        base = faiss.index_factory(d, "Flat", faiss.METRIC_INNER_PRODUCT)
        idx = faiss.IndexIDMap2(base)
        idx.add_with_ids(xb, np.arange(n, dtype=np.int64))
        logger.info(f"[FILTER] {label}: IVF build/train failed ({e}) -> FlatIP n={n} d={d}")
        return idx, 0

def _base_index(idx: faiss.Index) -> faiss.Index:
    return getattr(idx, "index", idx)


# --------------------------
# Session storage
# --------------------------
@dataclass
class SessionStore:
    tools: List[McpTool] = field(default_factory=list)
    actions_index: Optional[faiss.Index] = None
    entities_index: Optional[faiss.Index] = None
    entity_vecs: Optional[np.ndarray] = None
    created_at: float = field(default_factory=time.time)

_sessions: Dict[str, SessionStore] = {}

# --------------------------
# API models
# --------------------------
class IndexRequest(BaseModel):
    """
    EXACT input shape from MCP tools/list.
    We preserve tool objects 
    """
    tools: List[McpTool]
    session_id: Optional[str] = Field(
        default=None,
        description="Optional fixed id for the caller (host/client). If omitted, we generate one."
    )

class IndexResponse(BaseModel):
    session_id: str

class FilterRequest(BaseModel):
    """
    Query-time request: we only need the session_id and the user query.
    """
    session_id: str
    query: str = Field(..., description="User query to drive retrieval")


# --------------------------
# App & endpoints
# --------------------------
app = FastAPI(title="Filter Service (FAISS two-stage, sessioned)")

@app.get("/health")
def healthz():
    return {"ok": True, "sessions": len(_sessions)}


# --------------------------
# Dynamic-K selector: threshold + elbow (gap drop), clamped to [MIN_K, MAX_K]
# --------------------------
def pick_dynamic_k(scores: np.ndarray) -> int:
    if scores.size == 0:
        return 0

    keep_mask = (scores >= MIN_SCORE)
    k = int(keep_mask.sum())

    if k > 0:
        limit = min(len(scores), MAX_K)
        for i in range(1, limit):
            drop = float(scores[i-1] - scores[i])
            if drop >= GAP_DROP:
                k = min(k, i)
                break

    k = max(MIN_K, k)
    k = min(MAX_K, k, len(scores))
    return k


# ================================================================
# Indexing endpoint: store a session's tools and build indices
# ================================================================
@app.post("/index_tools", response_model=IndexResponse)
def index_tools(req: IndexRequest):
    tools_in: List[McpTool] = req.tools or []
    if not tools_in:
        raise HTTPException(status_code=400, detail="tools list is empty")

    names = [t.name or "" for t in tools_in]
    descs = [t.description or "" for t in tools_in]
    schemas = [t.inputSchema or {} for t in tools_in]

    actions_corpus, entities_corpus = [], []
    for name, desc, schema in zip(names, descs, schemas):
        a_text, e_text = extract_actions_entities_for_tool(name, desc, schema)
        actions_corpus.append(a_text or name)
        entities_corpus.append(e_text or desc or name)

    # MiniLM embeddings
    act = l2_normalize(embed_texts(actions_corpus))
    ent = l2_normalize(embed_texts(entities_corpus))

    actions_index, _ = _build_index(act, "actions")
    entities_index, _ = _build_index(ent, "entities")

    sid = req.session_id or f"sess-{int(time.time()*1000)}"
    _sessions[sid] = SessionStore(
        tools=tools_in,
        actions_index=actions_index,
        entities_index=entities_index,
        entity_vecs=ent,
    )
    logger.info(f"[FILTER] indexed session={sid} tools={len(tools_in)}")
    return IndexResponse(session_id=sid)


# ================================================================
# Filtering endpoint: require query + session_id, return EXACT ListToolsResult
# ================================================================
@app.post("/filter-tools", response_model=ListToolsResult)
def filter_tools(req: FilterRequest):
    """
    Endpoint handler:
    - Look up stored session by session_id (must be created via /index_tools).
    - Run two-stage retrieval (actions recall, entities re-rank) against
      the stored indices.
    - Return Top-K ORIGINAL tool objects in ranked order (K is dynamic).
    """
    sid = (req.session_id or "").strip()
    if not sid or sid not in _sessions:
        raise HTTPException(status_code=400, detail="unknown or missing session_id; call /index_tools first")

    store = _sessions[sid]
    tools_in: List[McpTool] = store.tools or []
    if not tools_in:
        raise HTTPException(status_code=400, detail="no tools are stored for this session")

    query = (req.query or "").strip()
    if not query:
        # When no query is supplied at runtime, return up to MAX_K unchanged
        return {"tools": tools_in[:MAX_K]}

    actions_index = store.actions_index
    ent_vecs = store.entity_vecs

    if actions_index is None or store.entities_index is None or ent_vecs is None:
        raise HTTPException(status_code=500, detail="indices not available for this session")

    # MiniLM query embedding
    q_vec = l2_normalize(embed_texts([query]))

    # Stage A: action similarity (recall) — search ALL candidates
    pool = len(tools_in)
    D, I = actions_index.search(q_vec, pool)
    mask = I[0] >= 0
    act_scores = D[0][mask].astype(np.float32)
    act_ids = I[0][mask].astype(np.int64)

    if act_ids.size == 0:
        return {"tools": tools_in[:MIN_K]}

    # Stage B: entity similarity (precision)
    ent_scores = ent_vecs[act_ids] @ q_vec[0].astype(np.float32)

    # Fuse and sort
    fused = (A_W * act_scores) + (E_W * ent_scores)
    order = np.argsort(-fused)
    sorted_ids = act_ids[order]
    sorted_scores = fused[order]

    # Dynamic-K selection
    k_final = pick_dynamic_k(sorted_scores)
    if k_final <= 0:
        k_final = min(MIN_K, len(sorted_ids))

    idxs = sorted_ids[:k_final].tolist()
    tools_out = [tools_in[i] for i in idxs]
    return {"tools": tools_out}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: pre-load the model
    logger.info("Pre-loading embedding model...")
    get_encoder()
    logger.info("Model loaded and ready!")
    yield
    # Shutdown: cleanup if needed
    logger.info("Shutting down...")


if __name__ == "__main__":
    import uvicorn

    # Pre-load the model
    logger.info("Pre-loading embedding model...")
    get_encoder()
    logger.info("Model loaded. Starting server...")

    # Run the server
    uvicorn.run(
        app,
        host="localhost",
        port=8000,
        log_level="info"
    )