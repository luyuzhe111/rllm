"""Shared fixtures for rllm-model-gateway tests."""

import json
import threading
import time
from typing import Any

import pytest
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from rllm_model_gateway import GatewayConfig, create_app
from rllm_model_gateway.store.memory_store import MemoryTraceStore

# ------------------------------------------------------------------
# Mock vLLM server
# ------------------------------------------------------------------

_MOCK_RESPONSE = {
    "id": "chatcmpl-mock",
    "object": "chat.completion",
    "model": "mock-model",
    "prompt_token_ids": [1, 2, 3, 4, 5],
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello from mock!"},
            "finish_reason": "stop",
            "stop_reason": None,
            "token_ids": [10, 11, 12],
            "logprobs": {
                "content": [
                    {"token": "Hello", "logprob": -0.5, "bytes": None, "top_logprobs": []},
                    {"token": " from", "logprob": -0.3, "bytes": None, "top_logprobs": []},
                    {"token": " mock!", "logprob": -0.1, "bytes": None, "top_logprobs": []},
                ]
            },
        }
    ],
    "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
    "prompt_logprobs": None,
    "kv_transfer_params": None,
}


def _build_mock_vllm_app() -> FastAPI:
    """Create a minimal mock vLLM server that returns canned responses."""
    app = FastAPI()
    app.state.request_log = []

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/v1/models")
    async def models():
        return {"data": [{"id": "mock-model", "object": "model"}]}

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        body = await request.json()
        app.state.request_log.append(body)

        if body.get("stream"):
            return StreamingResponse(_stream_chunks(), media_type="text/event-stream")
        return JSONResponse(content=_MOCK_RESPONSE)

    @app.post("/v1/completions")
    async def completions(request: Request):
        body = await request.json()
        app.state.request_log.append(body)
        return JSONResponse(content=_MOCK_RESPONSE)

    return app


def _stream_chunks():
    """Yield SSE chunks that mirror vLLM 0.11+ streaming format."""
    chunks = [
        {
            "id": "chatcmpl-mock",
            "object": "chat.completion.chunk",
            "model": "mock-model",
            "prompt_token_ids": [1, 2, 3, 4, 5],
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": ""},
                    "logprobs": None,
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "chatcmpl-mock",
            "object": "chat.completion.chunk",
            "model": "mock-model",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": "Hello"},
                    "logprobs": {"content": [{"token": "Hello", "logprob": -0.5, "bytes": None, "top_logprobs": []}]},
                    "finish_reason": None,
                    "token_ids": [10],
                }
            ],
        },
        {
            "id": "chatcmpl-mock",
            "object": "chat.completion.chunk",
            "model": "mock-model",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": " from mock!"},
                    "logprobs": {
                        "content": [
                            {"token": " from", "logprob": -0.3, "bytes": None, "top_logprobs": []},
                            {"token": " mock!", "logprob": -0.1, "bytes": None, "top_logprobs": []},
                        ]
                    },
                    "finish_reason": None,
                    "token_ids": [11, 12],
                }
            ],
        },
        {
            "id": "chatcmpl-mock",
            "object": "chat.completion.chunk",
            "model": "mock-model",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop", "stop_reason": None}],
            "usage": {
                "prompt_tokens": 5,
                "completion_tokens": 3,
                "total_tokens": 8,
            },
        },
    ]
    for chunk in chunks:
        yield f"data: {json.dumps(chunk)}\n\n"
    yield "data: [DONE]\n\n"


class MockVLLMServer:
    """Run a mock vLLM server in a background thread."""

    def __init__(self, host: str = "127.0.0.1", port: int = 0) -> None:
        self.host = host
        self.port = port
        self.app = _build_mock_vllm_app()
        self._server: uvicorn.Server | None = None
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def request_log(self) -> list[dict[str, Any]]:
        return self.app.state.request_log

    def start(self) -> None:
        config = uvicorn.Config(
            self.app,
            host=self.host,
            port=self.port,
            log_level="error",
        )
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._server.run, daemon=True)
        self._thread.start()
        # Wait for server to be ready
        deadline = time.time() + 5.0
        while time.time() < deadline:
            if self._server.started:
                # Grab the actual port if 0 was passed
                for sock in self._server.servers:
                    self.port = sock.sockets[0].getsockname()[1]
                return
            time.sleep(0.05)
        raise RuntimeError("Mock vLLM server failed to start")

    def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=5.0)


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture
def memory_store():
    return MemoryTraceStore()


@pytest.fixture
def mock_vllm():
    """Start a mock vLLM server and yield it."""
    server = MockVLLMServer(port=0)
    server.start()
    yield server
    server.stop()


@pytest.fixture
def gateway_config(mock_vllm: MockVLLMServer) -> GatewayConfig:
    """Config pointing at the mock vLLM server."""
    return GatewayConfig(
        port=0,
        store_worker="memory",
        workers=[{"url": f"{mock_vllm.url}/v1", "worker_id": "w0"}],
        health_check_interval=999,  # disable auto health checks in tests
    )


@pytest.fixture
def gateway_app(gateway_config: GatewayConfig) -> FastAPI:
    """Create a gateway FastAPI app (not running — use httpx TestClient)."""
    return create_app(gateway_config)
