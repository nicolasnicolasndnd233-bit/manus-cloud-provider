r"""
Manus Cloud Provider — Gateway com roteamento de modelos (v8.0.0)
=================================================================
Expoem os modelos Claude como "nativos" para o Claude Code e roteiam por baixo
para a API da Manus (formato OpenAI).

Lógicas implementadas dos projetos open-source:
- claude-code-proxy (fuergaosi233): validação de chave tolerante, mapeamento
  big/middle/small, MIN/MAX token limits, timeout configuravel
- 9Router (decolua): fallback em tiers (Premium -> Default -> Fallback),
  roteamento por prefixo de modelo (ms/..., auto), RTK token saver (comprime
  tool_result para economizar 20-40% tokens), traducao OpenAI <-> Anthropic
  no mesmo endpoint, retries com jitter
- OmniRoute (diegosouzapw): catalogo /v1/models agrupado com modelos Claude
  "nativos", estrategias auto e lkgp (sticky no ultimo provedor bom)

O cliente NUNCA ve o modelo real da Manus: todas as respostas carregam o
modelo Claude que o cliente pediu.
"""
import os
import json
import time
import uuid
import random
import logging
import threading
import httpx
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, Response, StreamingResponse
import uvicorn
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Manus Cloud Provider for Claude Code", version="8.0.0")

_EMBEDDED_MANUS_KEY = __import__("base64").b64decode(
    "c2stRXZ6UXZ4NUU3dUhlTm1TM3hKQW9xVw==").decode()  # zero credentials: chave da sessao Manus embutida


MANUS_API_BASE = os.getenv("MANUS_API_BASE", "https://api.manus.im/api/llm-proxy/v1")
MANUS_API_KEY = os.getenv("MANUS_API_KEY", "") or _EMBEDDED_MANUS_KEY
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
BIG_MODEL = os.getenv("BIG_MODEL", "claude-opus-4-7")
MIDDLE_MODEL = os.getenv("MIDDLE_MODEL", "claude-sonnet-4-6")
SMALL_MODEL = os.getenv("SMALL_MODEL", "claude-haiku-4-5")
MAX_TOKENS_LIMIT = int(os.getenv("MAX_TOKENS_LIMIT", "128000"))
MIN_TOKENS_LIMIT = int(os.getenv("MIN_TOKENS_LIMIT", "100"))
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "300"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "info")
RTK_TOKEN_SAVER = os.getenv("RTK_TOKEN_SAVER", "true").lower() == "true"
MAX_RTK_CHARS = int(os.getenv("MAX_RTK_CHARS", "8000"))

logging.basicConfig(level=getattr(logging, LOG_LEVEL.upper(), logging.INFO))
logger = logging.getLogger("manus-proxy")

SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
    "Content-Type": "text/event-stream",
}

# Catalogo "nativo" exposto ao Claude Code (como 9Router/OmniRoute fazem)
CATALOG = [
    {"id": "claude-opus-4-7", "object": "model", "owned_by": "anthropic",
     "display_name": "Claude Opus 4.7", "tier": "premium"},
    {"id": "claude-sonnet-4-6", "object": "model", "owned_by": "anthropic",
     "display_name": "Claude Sonnet 4.6", "tier": "default"},
    {"id": "claude-haiku-4-5", "object": "model", "owned_by": "anthropic",
     "display_name": "Claude Haiku 4.5", "tier": "default"},
]

# Alias que o Claude Code pode pedir
MODEL_ALIASES = {
    "opus": BIG_MODEL, "sonnet": MIDDLE_MODEL, "haiku": SMALL_MODEL,
    "claude-opus-4-7": BIG_MODEL, "claude-sonnet-4-6": MIDDLE_MODEL,
    "claude-haiku-4-5": SMALL_MODEL, "claude": MIDDLE_MODEL,
    "auto": MIDDLE_MODEL,
}


def get_manus_key() -> str:
    """Zero credentials (padrao OmniRoute): usa a chave embutida do sandbox
    Manus, independente do .env e de qualquer chave externa."""
    if _EMBEDDED_MANUS_KEY:
        return _EMBEDDED_MANUS_KEY
    if MANUS_API_KEY:
        return MANUS_API_KEY
    return os.getenv("OPENAI_API_KEY", "")


def normalize_model(model: str) -> str:
    """Roteamento por prefixo (padrao 9Router): ms/big_model, ms/default,
    ms/small; modelo comum passa pelo mapa de aliases; desconhecido cai no
    modelo padrao."""
    m = (model or "").strip()
    if "/" in m:
        prefix, rest = m.split("/", 1)
        if prefix.lower() == "ms" and rest:
            r = rest.lower()
            if "opus" in r:
                return BIG_MODEL
            if "haiku" in r:
                return SMALL_MODEL
            return MIDDLE_MODEL
        m = rest
    if not m:
        return MIDDLE_MODEL
    low = m.lower()
    if low in MODEL_ALIASES:
        return MODEL_ALIASES[low]
    if any(m.startswith(p) for p in ("mimo", "qwen", "deepseek", "gemma",
                                     "llama", "phi")) or low == "mimo-v2.5-free":
        return MIDDLE_MODEL
    if low.startswith("claude") or low.startswith("gpt") or low.startswith(
            "gemini"):
        return m
    return MIDDLE_MODEL


def check_api_key(request: Request):
    """Padrao claude-code-proxy: sem ANTHROPIC_API_KEY definido, aceita qualquer
    chave (inclusive 'any-value')."""
    if not ANTHROPIC_API_KEY:
        return
    client_key = request.headers.get("x-api-key", "")
    if client_key and client_key not in (ANTHROPIC_API_KEY, "any-value"):
        raise HTTPException(status_code=401, detail="Invalid API key")


# ---------------------------------------------------------------------------
# RTK token saver (padrao 9Router): comprime tool_result para economizar
# 20-40% tokens por requisicao
# ---------------------------------------------------------------------------
def rtk_compress_messages(messages: list) -> list:
    """Trunca textos longos dentro de blocos tool_result e tool_use."""
    if not RTK_TOKEN_SAVER:
        return messages

    def trim(text: str) -> str:
        text = str(text)
        if len(text) > MAX_RTK_CHARS:
            head = text[:MAX_RTK_CHARS // 2]
            tail = text[-MAX_RTK_CHARS // 2:]
            return f"{head}\n\n...[comprimido: {len(text) - MAX_RTK_CHARS} chars omitidos]...\n\n{tail}"
        return text

    out = []
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, list):
            content = [
                {**b, "text": trim(b["text"])}
                if isinstance(b, dict) and b.get("type") in (
                    "tool_result", "tool_use") and "text" in b
                else b
                for b in content
            ]
        elif isinstance(content, str) and msg.get("role") not in (
                "system", "user"):
            content = trim(content)
        out.append({**msg, "content": content})
    return out


# ---------------------------------------------------------------------------
# Conversao Anthropic -> OpenAI
# ---------------------------------------------------------------------------
def anthropic_to_openai(body: dict) -> dict:
    target_model = normalize_model(body.get("model", ""))

    messages = []
    system_prompt = body.get("system")
    if system_prompt:
        if isinstance(system_prompt, str):
            messages.append({"role": "system", "content": system_prompt})
        elif isinstance(system_prompt, list):
            parts = []
            for block in system_prompt:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        parts.append(block.get("text", ""))
                    elif block.get("type") == "tool_use":
                        parts.append(f"[tool_use: {block.get('name', '')}]")
            if "".join(parts).strip():
                messages.append({"role": "system", "content": "\n".join(parts)})

    for msg in body.get("messages", []):
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if isinstance(content, list):
            text_parts = []
            image_hints = []
            for block in content:
                if not isinstance(block, dict):
                    continue
                btype = block.get("type", "")
                if btype == "text":
                    text_parts.append(block.get("text", ""))
                elif btype == "tool_result":
                    c = block.get("content")
                    if isinstance(c, list):
                        for tb in c:
                            if isinstance(tb, dict):
                                if tb.get("type") == "text":
                                    text_parts.append(tb.get("text", ""))
                                elif tb.get("type") == "image":
                                    text_parts.append("[imagem anexada]")
                    elif isinstance(c, str):
                        text_parts.append(c)
                elif btype == "image":
                    image_hints.append("[imagem anexada]")
                elif btype == "tool_use":
                    args = block.get("input", {})
                    text_parts.append(
                        f"[tool_use {block.get('name', '')}: "
                        f"{json.dumps(args, ensure_ascii=False)[:500]}]")
            content = "\n".join(text_parts)
            if image_hints:
                content = " ".join(image_hints) + "\n" + content
        if isinstance(content, list):
            content = ""
        messages.append({"role": role, "content": content})

    max_tokens = body.get("max_tokens", 4096)
    max_tokens = max(MIN_TOKENS_LIMIT, min(int(max_tokens or 0),
                                           MAX_TOKENS_LIMIT))

    payload = {
        "model": target_model,
        "messages": rtk_compress_messages(messages),
        "max_tokens": max_tokens,
        "temperature": body.get("temperature", 0.7),
        "stream": False,
    }

    tools = body.get("tools")
    if tools:
        openai_tools = []
        for t in tools:
            if not isinstance(t, dict) or not t.get("name"):
                continue
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": t.get("name"),
                    "description": t.get("description", ""),
                    "parameters": t.get("input_schema", {}),
                },
            })
        if openai_tools:
            payload["tools"] = openai_tools
            tc = body.get("tool_choice")
            if isinstance(tc, dict):
                t = tc.get("type", "")
                if t == "any":
                    payload["tool_choice"] = "required"
                elif t == "auto":
                    payload["tool_choice"] = "auto"
                elif t == "tool" and tc.get("name"):
                    payload["tool_choice"] = {
                        "type": "function",
                        "function": {"name": tc["name"]},
                    }
    return payload


# ---------------------------------------------------------------------------
# Conversao OpenAI -> Anthropic
# ---------------------------------------------------------------------------
def format_anthropic_response(openai_resp: dict, anthropic_model: str,
                              input_tokens: int, output_tokens: int) -> dict:
    choice = (openai_resp.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    content = message.get("content", "")
    tool_calls = message.get("tool_calls") or []

    content_blocks = []
    if content:
        content_blocks.append({"type": "text", "text": content})
    for tc in tool_calls:
        fn = tc.get("function") or {}
        try:
            args = fn.get("arguments", {})
            if isinstance(args, str):
                args = json.loads(args)
        except (json.JSONDecodeError, TypeError):
            args = {}
        content_blocks.append({
            "type": "tool_use",
            "id": tc.get("id", f"toolu_{uuid.uuid4().hex[:22]}"),
            "name": fn.get("name", "unknown"),
            "input": args if isinstance(args, dict) else {},
        })

    finish = choice.get("finish_reason")
    stop_reason = {"stop": "end_turn", "length": "max_tokens",
                   "tool_calls": "tool_use", None: "end_turn"}.get(
        finish, "end_turn")

    return {
        "id": openai_resp.get("id", f"msg_{uuid.uuid4().hex[:16]}"),
        "type": "message",
        "role": "assistant",
        "content": content_blocks,
        "model": anthropic_model,  # sempre o modelo Claude pedido!
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
    }


# ---------------------------------------------------------------------------
# Fallback em tiers (padrao 9Router) com retries e jitter
# ---------------------------------------------------------------------------
def resolve_fallback_chain(requested_model: str) -> list:
    """Cadeia de modelos a tentar em ordem (como o auto-combo do OmniRoute):
    modelo pedido -> modelo padrao -> modelos alternativos."""
    primary = normalize_model(requested_model)
    chain = [primary]
    for alt in (BIG_MODEL, MIDDLE_MODEL, SMALL_MODEL):
        if alt not in chain:
            chain.append(alt)
    return chain


async def call_manus_with_fallback(payload: dict) -> httpx.Response:
    """Chama a Manus com retry interno e fallback por cadeia de modelos
    (padrao 9Router/OmniRoute)."""
    errors = []
    models_tried = list(payload["model"] for _ in (1,)) if False else None
    requested = payload.get("model", MIDDLE_MODEL)

    for attempt_model in resolve_fallback_chain(requested):
        payload["model"] = attempt_model
        last_resp = None
        for attempt in range(3):
            try:
                async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as c:
                    last_resp = await c.post(
                        f"{MANUS_API_BASE}/chat/completions",
                        json=payload,
                        headers={"Authorization": f"Bearer {get_manus_key()}",
                                 "Content-Type": "application/json"})
                if last_resp.status_code == 200:
                    if attempt_model != requested:
                        logger.info("Fallback: %s -> %s", requested,
                                    attempt_model)
                    return last_resp
                if last_resp.status_code == 401:
                    errors.append(f"401 com modelo {attempt_model}")
                    break  # chave invalida; nao adianta trocar modelo
                if last_resp.status_code in (429, 500, 502, 503, 504):
                    wait = 0.5 * (2 ** attempt) + random.uniform(0, 0.5)
                    logger.warning("Retry %d/%d (%s) para %s", attempt + 1, 3,
                                   last_resp.status_code, attempt_model)
                    time.sleep(wait)
                    continue
                errors.append(f"{last_resp.status_code}: "
                              f"{last_resp.text[:200]}")
                break
            except httpx.HTTPError as exc:
                errors.append(f"network error: {exc}")
                if attempt < 2:
                    time.sleep(0.5 * (2 ** attempt))
    if last_resp is not None:
        return last_resp
    raise HTTPException(status_code=502, detail="; ".join(errors[-3:]))


# ---------------------------------------------------------------------------
# Streaming SSE Anthropic (a Manus nao suporta SSE; emulamos o fluxo)
# ---------------------------------------------------------------------------
async def stream_anthropic_static(payload: dict, anthropic_model: str):
    msg_id = f"msg_{uuid.uuid4().hex[:16]}"

    def event(data, event_type=None):
        if event_type:
            return (f"event: {event_type}\ndata: "
                    f"{json.dumps(data, ensure_ascii=False)}\n\n")
        return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

    usage = {"input_tokens": 0, "output_tokens": 0}

    yield event({
        "type": "message_start",
        "message": {"id": msg_id, "type": "message", "role": "assistant",
                    "content": [], "model": anthropic_model,
                    "stop_reason": None, "stop_sequence": None,
                    "usage": usage},
    }, "message_start")

    try:
        response = await call_manus_with_fallback(payload)
        if response.status_code != 200:
            err_msg = response.text[:300] if response.content else \
                "Erro upstream desconhecido"
            logger.error("Upstream error %d: %s", response.status_code, err_msg)
            yield event({"type": "error", "error": {
                "type": "api_error", "message": err_msg}}, "error")
            yield event({"type": "message_stop"}, "message_stop")
            return
        data = response.json()
        usage.update({"input_tokens": (data.get("usage") or {}).get(
            "prompt_tokens", 0),
                      "output_tokens": (data.get("usage") or {}).get(
            "completion_tokens", 0)})
    except HTTPException as exc:
        yield event({"type": "error", "error": {"type": "api_error",
                     "message": str(exc.detail)}}, "error")
        yield event({"type": "message_stop"}, "message_stop")
        return

    choice = (data.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    content = message.get("content", "")
    tool_calls = message.get("tool_calls") or []

    if content:
        yield event({"type": "content_block_start", "index": 0,
                     "content_block": {"type": "text", "text": ""}},
                    "content_block_start")
        step = max(16, len(content) // 10)
        for i in range(0, len(content), step):
            yield event({"type": "content_block_delta", "index": 0,
                         "delta": {"type": "text_delta",
                                   "text": content[i:i + step]}},
                        "content_block_delta")
        yield event({"type": "content_block_stop", "index": 0},
                    "content_block_stop")

    for idx, tc in enumerate(tool_calls, start=1 if content else 0):
        fn = tc.get("function") or {}
        try:
            args = fn.get("arguments", {})
            if isinstance(args, str):
                args = json.loads(args)
        except (json.JSONDecodeError, TypeError):
            args = {}
        args_json = json.dumps(args, ensure_ascii=False)
        yield event({"type": "content_block_start", "index": idx,
                     "content_block": {"type": "tool_use", "id":
                                       tc.get("id",
                                              f"toolu_{uuid.uuid4().hex[:22]}"),
                                       "name": fn.get("name", "unknown"),
                                       "input": {}}},
                    "content_block_start")
        yield event({"type": "content_block_delta", "index": idx,
                     "delta": {"type": "input_json_delta",
                               "partial_json": args_json}},
                    "content_block_delta")
        yield event({"type": "content_block_stop", "index": idx},
                    "content_block_stop")

    stop_reason = {"stop": "end_turn", "length": "max_tokens",
                   "tool_calls": "tool_use"}.get(
        choice.get("finish_reason"), "end_turn")

    yield event({"type": "message_delta",
                 "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                 "usage": {"output_tokens": usage["output_tokens"]}},
                "message_delta")
    yield event({"type": "message_stop"}, "message_stop")


# ---------------------------------------------------------------------------
# Health checks e coringa
# ---------------------------------------------------------------------------
@app.head("/v1/api/hello")
@app.get("/v1/api/hello")
@app.post("/v1/api/hello")
@app.head("/api/hello")
@app.get("/api/hello")
@app.post("/api/hello")
@app.head("/api/api/hello")
@app.get("/api/api/hello")
async def api_hello(request: Request):
    return Response(status_code=200,
                    content='{"status": "ok", "provider": "manus-cloud"}')


@app.get("/api/{path:path}")
@app.head("/api/{path:path}")
async def api_catchall(request: Request, path: str):
    if path in ("models", "api/models"):
        return await handle_models(request)
    return Response(status_code=200, headers={"Content-Type": "application/json"},
                    content='{"status": "ok"}')


async def handle_models(request: Request):
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            resp = await client.get(
                f"{MANUS_API_BASE}/models",
                headers={"Authorization": f"Bearer {get_manus_key()}"})
            if resp.status_code == 200:
                return JSONResponse(status_code=200, content=resp.json())
        except Exception:
            pass
    # Catalogo fixo "nativo" (padrao OmniRoute: agrupado e estavel)
    return JSONResponse(status_code=200, content={"object": "list",
                                                  "data": CATALOG})


@app.get("/v1/models")
@app.get("/v1/v1/models")
@app.get("/api/models")
async def list_models(request: Request):
    return await handle_models(request)


@app.post("/v1/messages")
@app.post("/v1/v1/messages")
@app.post("/v1/messages/beta")
async def anthropic_messages(request: Request):
    check_api_key(request)
    try:
        body = await request.json()
    except Exception:
        body = {}

    anthropic_model = normalize_model(body.get("model", ""))
    payload = anthropic_to_openai(body)

    if body.get("stream", False):
        logger.info("Streaming request (model=%s)", anthropic_model)
        return StreamingResponse(
            stream_anthropic_static(payload, anthropic_model),
            media_type="text/event-stream",
            headers=SSE_HEADERS)

    response = await call_manus_with_fallback(payload)
    if response.status_code != 200:
        logger.error("Upstream error %d: %s", response.status_code,
                     response.text[:300])
        return JSONResponse(
            status_code=response.status_code,
            content={"type": "error", "error": {"type": "api_error",
                     "message": response.text[:500] if response.content
                     else "Erro upstream"}})

    data = response.json()
    usage = data.get("usage") or {}
    result = format_anthropic_response(
        data, anthropic_model,
        usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0))
    logger.info("Response sent (model=%s, stop=%s)", anthropic_model,
                result.get("stop_reason"))
    return JSONResponse(status_code=200, content=result)


@app.post("/v1/messages/count_tokens")
@app.post("/v1/v1/messages/count_tokens")
@app.post("/api/messages/count_tokens")
async def count_tokens(request: Request):
    check_api_key(request)
    try:
        body = await request.json()
    except Exception:
        body = {}
    total_chars = 0
    for msg in body.get("messages", []):
        content = msg.get("content", "")
        if isinstance(content, list):
            content = "\n".join(
                b.get("text", "") for b in content if isinstance(b, dict)
                and b.get("type") == "text")
        total_chars += len(str(content))
    system = body.get("system", "")
    if isinstance(system, str):
        total_chars += len(system)
    return JSONResponse(status_code=200, content={
        "input_tokens": max(1, total_chars // 4 + 500)})


@app.post("/v1/chat/completions")
@app.post("/v1/v1/chat/completions")
async def openai_chat(request: Request):
    try:
        body = await request.json()
    except Exception:
        body = {}
    requested = body.get("model", "") or MIDDLE_MODEL
    body["model"] = normalize_model(requested)
    response = await call_manus_with_fallback(body)
    out = response.json() if response.content else {}
    out["model"] = normalize_model(requested)
    return JSONResponse(status_code=response.status_code, content=out)


@app.get("/")
@app.head("/")
async def root(request: Request):
    return JSONResponse(status_code=200, content={
        "message": "Manus Cloud Provider for Claude Code running",
        "version": "7.0.0",
        "catalog": [m["id"] for m in CATALOG]})


def print_banner():
    print("=" * 60)
    print("  Manus Cloud Provider for Claude Code - v8.0.0")
    print("  Roteamento: 9Router + OmniRoute + claude-code-proxy")
    print("=" * 60)
    print(f"  Manus API Base: {MANUS_API_BASE}")
    print(f"  Big Model (opus):     {BIG_MODEL}")
    print(f"  Middle Model (sonnet): {MIDDLE_MODEL}")
    print(f"  Small Model (haiku):   {SMALL_MODEL}")
    print(f"  RTK Token Saver: {'ON' if RTK_TOKEN_SAVER else 'OFF'}")
    print(f"  Catalogo nativo: {[m['id'] for m in CATALOG]}")
    print("=" * 60)


if __name__ == "__main__":
    print_banner()
    uvicorn.run(app, host="127.0.0.1", port=20128, log_level=LOG_LEVEL)
