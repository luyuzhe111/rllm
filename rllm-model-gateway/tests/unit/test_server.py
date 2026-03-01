"""Integration tests for the gateway server (FastAPI app).

Uses httpx.AsyncClient as a test client against the app, with a real
mock vLLM backend server for proxy tests.
"""

import json

import httpx
import pytest
import pytest_asyncio
from rllm_model_gateway import GatewayConfig, create_app

from .conftest import MockVLLMServer

# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture
def mock_vllm():
    server = MockVLLMServer(port=0)
    server.start()
    yield server
    server.stop()


@pytest.fixture
def app(mock_vllm: MockVLLMServer):
    config = GatewayConfig(
        store_worker="memory",
        workers=[{"url": f"{mock_vllm.url}/v1", "worker_id": "w0"}],
        health_check_interval=999,
    )
    return create_app(config)


@pytest_asyncio.fixture
async def client(app):
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://testserver",
    ) as c:
        yield c


# ------------------------------------------------------------------
# Health endpoints
# ------------------------------------------------------------------


class TestHealth:
    @pytest.mark.asyncio
    async def test_health(self, client: httpx.AsyncClient):
        resp = await client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    @pytest.mark.asyncio
    async def test_health_workers(self, client: httpx.AsyncClient):
        resp = await client.get("/health/workers")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] >= 1


# ------------------------------------------------------------------
# Session management
# ------------------------------------------------------------------


class TestSessions:
    @pytest.mark.asyncio
    async def test_create_session(self, client: httpx.AsyncClient):
        resp = await client.post("/sessions", json={})
        assert resp.status_code == 200
        data = resp.json()
        assert "session_id" in data
        assert "url" in data

    @pytest.mark.asyncio
    async def test_create_session_with_id(self, client: httpx.AsyncClient):
        resp = await client.post("/sessions", json={"session_id": "my-session"})
        assert resp.status_code == 200
        assert resp.json()["session_id"] == "my-session"

    @pytest.mark.asyncio
    async def test_get_session_not_found(self, client: httpx.AsyncClient):
        resp = await client.get("/sessions/nonexistent")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_list_sessions_empty(self, client: httpx.AsyncClient):
        resp = await client.get("/sessions")
        assert resp.status_code == 200
        assert resp.json() == []

    @pytest.mark.asyncio
    async def test_delete_session(self, client: httpx.AsyncClient):
        await client.post("/sessions", json={"session_id": "to-delete"})
        resp = await client.delete("/sessions/to-delete")
        assert resp.status_code == 200


# ------------------------------------------------------------------
# Admin / worker management
# ------------------------------------------------------------------


class TestAdmin:
    @pytest.mark.asyncio
    async def test_list_workers(self, client: httpx.AsyncClient):
        resp = await client.get("/admin/workers")
        assert resp.status_code == 200
        workers = resp.json()
        assert len(workers) >= 1
        assert workers[0]["worker_id"] == "w0"

    @pytest.mark.asyncio
    async def test_add_worker(self, client: httpx.AsyncClient):
        resp = await client.post(
            "/admin/workers",
            json={"url": "http://new-worker:8000/v1"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "worker_id" in data

        # Verify it shows up in list
        workers = await client.get("/admin/workers")
        urls = [w["url"] for w in workers.json()]
        assert "http://new-worker:8000/v1" in urls

    @pytest.mark.asyncio
    async def test_add_worker_missing_url(self, client: httpx.AsyncClient):
        resp = await client.post("/admin/workers", json={})
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_remove_worker(self, client: httpx.AsyncClient):
        add_resp = await client.post(
            "/admin/workers",
            json={"url": "http://temp:8000/v1"},
        )
        wid = add_resp.json()["worker_id"]
        resp = await client.delete(f"/admin/workers/{wid}")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_remove_nonexistent_worker(self, client: httpx.AsyncClient):
        resp = await client.delete("/admin/workers/nonexistent")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_flush(self, client: httpx.AsyncClient):
        resp = await client.post("/admin/flush")
        assert resp.status_code == 200
        assert resp.json()["status"] == "flushed"


# ------------------------------------------------------------------
# Proxy (non-streaming)
# ------------------------------------------------------------------


class TestProxy:
    @pytest.mark.asyncio
    async def test_chat_completions_bare(self, client: httpx.AsyncClient):
        """POST /v1/chat/completions without session — should proxy successfully."""
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "mock-model",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "choices" in data
        # vLLM fields should be stripped
        assert "prompt_token_ids" not in data
        assert "prompt_logprobs" not in data
        choice = data.get("choices", [{}])[0]
        assert "token_ids" not in choice
        assert "stop_reason" not in choice

    @pytest.mark.asyncio
    async def test_chat_completions_with_session(self, client: httpx.AsyncClient):
        """POST /sessions/{sid}/v1/chat/completions — should proxy and capture trace."""
        resp = await client.post(
            "/sessions/test-sess/v1/chat/completions",
            json={
                "model": "mock-model",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["choices"][0]["message"]["content"] == "Hello from mock!"

    @pytest.mark.asyncio
    async def test_models_endpoint(self, client: httpx.AsyncClient):
        """GET /v1/models should proxy to the worker."""
        resp = await client.get("/v1/models")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_logprobs_injected(self, client: httpx.AsyncClient, mock_vllm: MockVLLMServer):
        """Middleware should inject logprobs=True into the request body."""
        await client.post(
            "/v1/chat/completions",
            json={
                "model": "mock-model",
                "messages": [{"role": "user", "content": "test"}],
            },
        )
        assert len(mock_vllm.request_log) >= 1
        last_req = mock_vllm.request_log[-1]
        assert last_req.get("logprobs") is True
        assert last_req.get("return_token_ids") is True


# ------------------------------------------------------------------
# Proxy (streaming)
# ------------------------------------------------------------------


class TestStreamingProxy:
    @pytest.mark.asyncio
    async def test_streaming_strips_vllm_fields(self, client: httpx.AsyncClient):
        """SSE chunks returned to agent must NOT contain vLLM-specific fields."""
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "mock-model",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        )
        assert resp.status_code == 200

        lines = [line for line in resp.text.strip().split("\n") if line.startswith("data: ") and line.strip() != "data: [DONE]"]
        assert len(lines) >= 1

        for line in lines:
            chunk = json.loads(line[6:])
            # Root-level vLLM fields must be stripped
            assert "prompt_token_ids" not in chunk
            assert "prompt_logprobs" not in chunk
            assert "kv_transfer_params" not in chunk
            # Choice-level vLLM fields must be stripped
            for choice in chunk.get("choices", []):
                assert "token_ids" not in choice
                assert "stop_reason" not in choice

    @pytest.mark.asyncio
    async def test_streaming_content_intact(self, client: httpx.AsyncClient):
        """SSE streaming should deliver all content from the mock."""
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "mock-model",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        )
        assert resp.status_code == 200

        content_parts = []
        for line in resp.text.strip().split("\n"):
            if line.startswith("data: ") and line.strip() != "data: [DONE]":
                chunk = json.loads(line[6:])
                for choice in chunk.get("choices", []):
                    c = choice.get("delta", {}).get("content")
                    if c:
                        content_parts.append(c)
        assert "".join(content_parts) == "Hello from mock!"

    @pytest.mark.asyncio
    async def test_streaming_trace_captured(self, mock_vllm: MockVLLMServer):
        """Streaming call with session should capture a trace with token data."""
        config = GatewayConfig(
            store_worker="memory",
            workers=[{"url": f"{mock_vllm.url}/v1", "worker_id": "w0"}],
            health_check_interval=999,
            sync_traces=True,
        )
        app = create_app(config)

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://testserver",
        ) as client:
            resp = await client.post(
                "/sessions/stream-test/v1/chat/completions",
                json={
                    "model": "mock-model",
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": True,
                },
            )
            assert resp.status_code == 200

            traces_resp = await client.get("/sessions/stream-test/traces")
            traces = traces_resp.json()
            assert len(traces) == 1

            trace = traces[0]
            assert trace["session_id"] == "stream-test"
            assert trace["prompt_token_ids"] == [1, 2, 3, 4, 5]
            assert trace["completion_token_ids"] == [10, 11, 12]
            assert trace["logprobs"] == [-0.5, -0.3, -0.1]
            assert trace["response_message"]["role"] == "assistant"
            assert "Hello" in trace["response_message"]["content"]


# ------------------------------------------------------------------
# Trace capture and retrieval
# ------------------------------------------------------------------


class TestTraceCapture:
    @pytest.mark.asyncio
    async def test_trace_persisted_for_session(self, mock_vllm: MockVLLMServer):
        """After a proxied call with session, traces should be retrievable."""
        config = GatewayConfig(
            store_worker="memory",
            workers=[{"url": f"{mock_vllm.url}/v1", "worker_id": "w0"}],
            health_check_interval=999,
            sync_traces=True,
        )
        app = create_app(config)

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://testserver",
        ) as client:
            resp = await client.post(
                "/sessions/trace-test/v1/chat/completions",
                json={
                    "model": "mock-model",
                    "messages": [{"role": "user", "content": "hello"}],
                },
            )
            assert resp.status_code == 200

            traces_resp = await client.get("/sessions/trace-test/traces")
            assert traces_resp.status_code == 200
            traces = traces_resp.json()
            assert len(traces) == 1

            trace = traces[0]
            assert trace["session_id"] == "trace-test"
            assert trace["prompt_token_ids"] == [1, 2, 3, 4, 5]
            assert trace["completion_token_ids"] == [10, 11, 12]
            assert trace["logprobs"] == [-0.5, -0.3, -0.1]
            assert trace["finish_reason"] == "stop"

    @pytest.mark.asyncio
    async def test_get_trace_by_id(self, mock_vllm: MockVLLMServer):
        config = GatewayConfig(
            store_worker="memory",
            workers=[{"url": f"{mock_vllm.url}/v1", "worker_id": "w0"}],
            health_check_interval=999,
            sync_traces=True,
        )
        app = create_app(config)

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://testserver",
        ) as client:
            await client.post(
                "/sessions/s1/v1/chat/completions",
                json={
                    "model": "mock-model",
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )

            traces = (await client.get("/sessions/s1/traces")).json()
            trace_id = traces[0]["trace_id"]

            resp = await client.get(f"/traces/{trace_id}")
            assert resp.status_code == 200
            assert resp.json()["trace_id"] == trace_id

    @pytest.mark.asyncio
    async def test_get_trace_not_found(self, client: httpx.AsyncClient):
        resp = await client.get("/traces/nonexistent")
        assert resp.status_code == 404
