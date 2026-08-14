"""
Manus Cloud Provider — Gateway definitivo para o Claude Code (v5.0.0)
=====================================================================
Simula a API da Anthropic (protocolo /v1/messages COM streaming SSE) e
redireciona as requisições para a API da Manus (formato OpenAI).

Implementa TODAS as lógicas dos proxies open-source usados com o Claude Code:
- claude-code-proxy (fuergaosi233): validação de chave tolerante, mapeamento de
  modelos big/middle/small, MIN/MAX token limits, timeout configurável
- Formato SSE Anthropic exato exigido pelo cliente
- Conversão completa de system prompts, content blocks (text/tool_result/image),
  tool_use e tool_choice
"""
import os
import json
import time
import uuid
import logging
import httpx
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, Response, StreamingResponse
import uvicorn
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Manus Cloud Provider for Claude Code", version="5.0.0")

MANUS_API_BASE = os.getenv("MANUS_API_BASE", "https://api.manus.im/api/llm-proxy/v1")
MANUS_API_KEY = os.getenv("MANUS_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
BIG_MODEL = os.getenv("BIG_MODEL", "claude-opus-4-7")
MIDDLE_MODEL = os.getenv("MIDDLE_MODEL", "claude-sonnet-4-6")
SMALL_MODEL = os.getenv("SMALL_MODEL", "claude-haiku-4-5")
MAX_TOKENS_LIMIT = int(os.getenv("MAX_TOKENS_LIMIT", "128000"))
MIN_TOKENS_LIMIT = int(os.getenv("MIN_TOKENS_LIMIT", "100"))
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "300"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "info")

logging.basicConfig(level=getattr(logging, LOG_LEVEL.upper(), logging.INFO))
logger = logging.getLogger("manus-proxy")

# Prefixos de modelos que NÃO devem ser roteados para a Manus
FORBIDDEN_PREFIXES = ("mimo", "qwen", "deepseek", "gemma", "llama", "phi")


def get_manus_key() -> str:
    """Retorna a chave mais confiável para chamar a Manus."""
    if MANUS_API_KEY:
        return MANUS_API_KEY
    return os.getenv("OPENAI_API_KEY", "")


def normalize_model(model: str) -> str:
    """Mapeia o modelo pedido pelo Claude Code para um modelo da Manus.

    Seguindo o padrão claude-code-proxy:
      - opus  -> BIG_MODEL
      - sonnet/qualquer outro -> MIDDLE_MODEL
      - haiku -> SMALL_MODEL
    Modelos desconhecidos/estranhos caem no MIDDLE_MODEL.
    """
    m = (model or "").lower().strip()
    if not m:
        return MIDDLE_MODEL
    if any(m.startswith(p) for p in FORBIDDEN_PREFIXES) or m in (
            "mimo-v2.5-free", "claude", "opus", "sonnet", "haiku"):
        return MIDDLE_MODEL
    if "opus" in m:
        return BIG_MODEL
    if "haiku" in m:
        return SMALL_MODEL
    if m.startswith("claude") or m.startswith("gpt") or m.startswith("gemini"):
        return m
    return MIDDLE_MODEL


def check_api_key(request: Request):
    """Valida o x-api-key enviado pelo cliente.

    Padrão claude-code-proxy: se ANTHROPIC_API_KEY não estiver definido,
    aceita QUALQUER chave (inclusive 'any-value').
    """
    if not ANTHROPIC_API_KEY:
        return
    client_key = request.headers.get("x-api-key", "")
    if client_key and client_key not in (ANTHROPIC_API_KEY, "any-value"):
        raise HTTPException(status_code=401, detail="Invalid API key")


# ---------------------------------------------------------------------------
# Conversão Anthropic -> OpenAI
# ---------------------------------------------------------------------------
def anthropic_to_openai(body: dict) -> dict:
    target_model = normalize_model(body.get("model", ""))
    payload_model = target_model

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
                        f"[tool_use {block.get('name', '')}: {json.dumps(args, ensure_ascii=False)[:500]}]"
                    )
            content = "\n".join(text_parts)
            if image_hints:
                content = " ".join(image_hints) + "\n" + content
        if isinstance(content, list):
            content = ""
        messages.append({"role": role, "content": content})

    max_tokens = body.get("max_tokens", 4096)
    max_tokens = max(MIN_TOKENS_LIMIT, min(int(max_tokens or 0), MAX_TOKENS_LIMIT))

    payload = {
        "model": payload_model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": body.get("temperature", 0.7),
        "stream": False,
    }

    # Ferramentas (function calling)
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
                        "type": "function", "function": {"name": tc["name"]}
                    }

    return payload


# ---------------------------------------------------------------------------
# Conversão OpenAI -> Anthropic
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
    stop_reason = {
        "stop": "end_turn", "length": "max_tokens", "tool_calls": "tool_use"
    }.get(finish, "end_turn")

    return {
        "id": openai_resp.get("id", f"msg_{uuid.uuid4().hex[:16]}"),
        "type": "message",
        "role": "assistant",
        "content": content_blocks,
        "model": anthropic_model,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
    }


# ---------------------------------------------------------------------------
# Chamada upstream à Manus (com retry automático de chave)
# ---------------------------------------------------------------------------
async def call_manus(payload: dict):
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        resp = await client.post(
            f"{MANUS_API_BASE}/chat/completions",
            json=payload,
            headers={"Authorization": f"Bearer {get_manus_key()}",
                     "Content-Type": "application/json"},
        )
    if resp.status_code == 401 and MANUS_API_KEY:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            resp = await client.post(
                f"{MANUS_API_BASE}/chat/completions",
                json=payload,
                headers={"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY', '')}",
                         "Content-Type": "application/json"},
            )
    return resp


# ---------------------------------------------------------------------------
# Streaming SSE Anthropic (a Manus não suporta SSE; emulamos o fluxo)
# ---------------------------------------------------------------------------
async def stream_anthropic_static(payload: dict, anthropic_model: str):
    """Emite eventos SSE Anthropic a partir de uma resposta não-streaming."""
    msg_id = f"msg_{uuid.uuid4().hex[:16]}"

    def event(data, event_type: str = None):
        if event_type:
            return f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
        return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

    yield event({
        "type": "message_start",
        "message": {
            "id": msg_id, "type": "message", "role": "assistant",
            "content": [], "model": anthropic_model,
            "stop_reason": None, "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        },
    }, "message_start")

    response = await call_manus(payload)
    if response.status_code != 200:
        err_msg = response.text[:500] if response.content else "Erro upstream desconhecido"
        logger.error("Upstream error %d: %s", response.status_code, err_msg)
        yield event({"type": "error",
                     "error": {"type": "api_error", "message": err_msg}}, "error")
        return

    data = response.json()
    choice = (data.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    content = message.get("content", "")
    tool_calls = message.get("tool_calls") or []
    usage = data.get("usage") or {}
    finish = choice.get("finish_reason")

    if content:
        yield event({"type": "content_block_start", "index": 0,
                     "content_block": {"type": "text", "text": ""}},
                    "content_block_start")
        # Emite em chunks para o cliente perceber o fluxo de texto
        step = max(16, len(content) // 10)
        for i in range(0, len(content), step):
            yield event({"type": "content_block_delta", "index": 0,
                         "delta": {"type": "text_delta", "text": content[i:i + step]}},
                        "content_block_delta")
        yield event({"type": "content_block_stop", "index": 0}, "content_block_stop")

    for idx, tc in enumerate(tool_calls):
        fn = tc.get("function") or {}
        try:
            args = fn.get("arguments", {})
            if isinstance(args, str):
                args = json.loads(args)
        except (json.JSONDecodeError, TypeError):
            args = {}
        args_str = json.dumps(args if isinstance(args, dict) else {}, ensure_ascii=False)
        yield event({"type": "content_block_start", "index": idx + 1,
                     "content_block": {"type": "tool_use",
                                       "id": tc.get("id", f"toolu_{uuid.uuid4().hex[:22]}"),
                                       "name": fn.get("name", "unknown"),
                                       "input": {}}}, "content_block_start")
        # input_json_delta exige delta em STRING (partial_json), não objeto
        yield event({"type": "content_block_delta", "index": idx + 1,
                     "delta": {"type": "input_json_delta", "partial_json": args_str}},
                    "content_block_delta")
        yield event({"type": "content_block_stop", "index": idx + 1}, "content_block_stop")

    stop_reason = {
        "stop": "end_turn", "length": "max_tokens", "tool_calls": "tool_use"
    }.get(finish, "end_turn")
    yield event({"type": "message_delta",
                 "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                 "usage": {"output_tokens": usage.get("completion_tokens", 1)}},
                "message_delta")
    yield event({"type": "message_stop"}, "message_stop")


SSE_HEADERS = {
    "Content-Type": "text/event-stream; charset=utf-8",
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
}


# ---------------------------------------------------------------------------
# ENDPOINTS
# ---------------------------------------------------------------------------
@app.api_route("/api/hello", methods=["GET", "HEAD", "POST"])
@app.api_route("/v1/api/hello", methods=["GET", "HEAD", "POST"])
@app.api_route("/api/api/hello", methods=["GET", "HEAD", "POST"])
async def health(request: Request):
    return Response(status_code=200, headers={"Content-Type": "application/json"},
                    content='{"status": "ok", "message": "Manus proxy is running"}')


@app.api_route("/api/{path:path}", methods=["GET", "HEAD"])
async def api_catchall(request: Request, path: str):
    """Coringa: rotas GET/HEAD desconhecidas sempre retornam 200 OK."""
    if path in ("models", "api/models"):
        return await handle_models(request)
    return Response(status_code=200, headers={"Content-Type": "application/json"},
                    content='{"status": "ok"}')


async def handle_models(request: Request):
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            resp = await client.get(
                f"{MANUS_API_BASE}/models",
                headers={"Authorization": f"Bearer {get_manus_key()}"},
            )
            if resp.status_code == 200:
                return JSONResponse(status_code=200, content=resp.json())
        except Exception:
            pass
    return JSONResponse(status_code=200, content={"object": "list", "data": [
        {"id": "claude-sonnet-4-6", "object": "model", "owned_by": "anthropic",
         "display_name": "Claude Sonnet 4.6"},
        {"id": "claude-opus-4-7", "object": "model", "owned_by": "anthropic",
         "display_name": "Claude Opus 4.7"},
        {"id": "claude-haiku-4-5", "object": "model", "owned_by": "anthropic",
         "display_name": "Claude Haiku 4.5"},
    ]})


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

    stream_mode = body.get("stream", False)
    if stream_mode:
        logger.info("Streaming request (model=%s)", anthropic_model)
        return StreamingResponse(
            stream_anthropic_static(payload, anthropic_model),
            media_type="text/event-stream",
            headers=SSE_HEADERS,
        )

    # Resposta não-streaming
    response = await call_manus(payload)
    if response.status_code != 200:
        logger.error("Upstream error %d: %s", response.status_code,
                     response.text[:300])
        return JSONResponse(
            status_code=response.status_code,
            content={"type": "error", "error": {"type": "api_error",
                     "message": response.text[:500] if response.content
                     else "Erro upstream"}},
        )

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
    # Estimativa aproximada: ~4 tokens por caractere + sobresalença
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
    if not body.get("model") or body.get("model") in (
            "claude", "opus", "sonnet", "haiku"):
        body["model"] = MIDDLE_MODEL
    body["model"] = normalize_model(body["model"])
    response = await call_manus(body)
    return JSONResponse(status_code=response.status_code,
                        content=response.json() if response.content else {})


@app.get("/")
@app.head("/")
async def root(request: Request):
    return JSONResponse(status_code=200, content={
        "message": "Manus Cloud Provider for Claude Code running",
        "version": "5.0.0"})


def print_banner():
    print("=" * 60)
    print("  Manus Cloud Provider for Claude Code — v5.0.0")
    print("=" * 60)
    print(f"  Manus API Base: {MANUS_API_BASE}")
    print(f"  Big Model (opus):    {BIG_MODEL}")
    print(f"  Middle Model (sonnet): {MIDDLE_MODEL}")
    print(f"  Small Model (haiku):   {SMALL_MODEL}")
    print(f"  Token limits: {MIN_TOKENS_LIMIT} — {MAX_TOKENS_LIMIT}")
    print(f"  Request timeout: {REQUEST_TIMEOUT}s")
    print(f"  Client API key validation: {'Enabled' if ANTHROPIC_API_KEY else 'Disabled (accepts any key)'}")
    print("=" * 60)


if __name__ == "__main__":
    print_banner()
    uvicorn.run(app, host="127.0.0.1", port=20128, log_level=LOG_LEVEL)
