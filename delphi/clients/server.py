"""HTTP client for externally hosted OpenAI-compatible chat servers."""

from dataclasses import dataclass

import httpx

from .client import Client, Response


@dataclass
class TopLogprob:
    token: str
    logprob: float


@dataclass
class TokenLogprobs:
    token: str
    top_logprobs: list[TopLogprob]


class ServerClient(Client):
    """Reuse an external server without importing or initializing vLLM locally."""

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str | None = None,
        max_tokens: int = 500,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        super().__init__(model)
        self.max_tokens = max_tokens
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.http = httpx.AsyncClient(
            base_url=base_url.rstrip("/") + "/",
            headers=headers,
            timeout=120.0,
            transport=transport,
        )

    async def generate(self, prompt: str | list[dict[str, str]], **kwargs) -> Response:
        if "prompt_logprobs" in kwargs:
            raise ValueError(
                "Chat servers do not provide prompt logprobs; use JSON simulation"
            )
        if kwargs.get("stream"):
            raise ValueError("Streaming responses are not supported")
        messages = (
            prompt
            if isinstance(prompt, list)
            else [{"role": "user", "content": prompt}]
        )
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": self.max_tokens,
            **kwargs,
        }
        response = await self.http.post("chat/completions", json=payload)
        response.raise_for_status()
        choice = response.json()["choices"][0]
        entries = (choice.get("logprobs") or {}).get("content")
        logprobs = (
            None
            if entries is None
            else [
                TokenLogprobs(
                    token=entry["token"],
                    top_logprobs=[
                        TopLogprob(token=p["token"], logprob=float(p["logprob"]))
                        for p in entry.get("top_logprobs", [])
                    ],
                )
                for entry in entries
            ]
        )
        return Response(text=choice["message"].get("content") or "", logprobs=logprobs)

    async def close(self):
        await self.http.aclose()
