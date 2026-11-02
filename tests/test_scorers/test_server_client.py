import json
import subprocess
import sys

import httpx
import pytest

from delphi.clients.server import ServerClient
from delphi.scorers.classifier.detection import DetectionScorer
from delphi.scorers.simulator.simulation.simulator import NeuronSimulator


def reply(content="[1]", **choice):
    return {"choices": [{"message": {"content": content}, **choice}]}


@pytest.mark.asyncio
async def test_remote_url_parameters_auth_and_logprobs():
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(
            200,
            json=reply(
                logprobs={
                    "content": [
                        {
                            "token": "1",
                            "top_logprobs": [
                                {"token": "0", "logprob": -2.0},
                                {"token": "1", "logprob": -0.2},
                            ],
                        }
                    ]
                }
            ),
        )

    client = ServerClient(
        "model",
        "https://remote.example/v1",
        api_key="test",
        transport=httpx.MockTransport(handler),
    )
    try:
        response = await client.generate(
            "hello", seed=42, top_p=0.8, stop=["end"], logprobs=True, top_logprobs=2
        )
        request = seen[0]
        assert str(request.url) == "https://remote.example/v1/chat/completions"
        assert request.headers["Authorization"] == "Bearer test"
        body = json.loads(request.content)
        assert body["messages"] == [{"role": "user", "content": "hello"}]
        assert body["seed"] == 42 and body["stop"] == ["end"] and body["top_p"] == 0.8
        predictions, probabilities = DetectionScorer(client)._parse(
            response.text, response.logprobs
        )
        assert predictions == [True] and probabilities[0] > 0.8
    finally:
        await client.close()
    assert client.http.is_closed


@pytest.mark.asyncio
async def test_server_simulation_uses_json_and_returns_nonzero_activations():
    requests = []

    def handler(request):
        body = json.loads(request.content)
        requests.append(body)
        assert "prompt_logprobs" not in body
        return httpx.Response(
            200,
            json=reply(
                json.dumps(
                    {
                        "activations": [{"token": "cat", "activation": 7}],
                    }
                )
            ),
        )

    client = ServerClient(
        "model", "http://localhost:8000/v1/", transport=httpx.MockTransport(handler)
    )
    try:
        simulator = NeuronSimulator(
            client, "cats", use_logprobs=client.supports_prompt_logprobs
        )
        result = await simulator.simulate(["cat"])
        assert result.expected_activations == [7.0]
        assert len(requests) == 1
    finally:
        await client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("kwargs", [{"prompt_logprobs": 5}, {"stream": True}])
async def test_unsupported_requests_fail_explicitly(kwargs):
    client = ServerClient(
        "model",
        "http://localhost/v1/",
        transport=httpx.MockTransport(
            lambda request: pytest.fail("Invalid requests must not reach the server")
        ),
    )
    try:
        with pytest.raises(ValueError):
            await client.generate("hello", **kwargs)
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_http_errors_propagate():
    client = ServerClient(
        "model",
        "http://localhost/v1/",
        transport=httpx.MockTransport(
            lambda request: httpx.Response(503, json={"error": "unavailable"})
        ),
    )
    try:
        with pytest.raises(httpx.HTTPStatusError):
            await client.generate("hello")
    finally:
        await client.close()


def test_importing_server_client_does_not_import_vllm():
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; from delphi.clients import ServerClient; "
                'assert "vllm" not in sys.modules'
            ),
        ],
        check=True,
    )
