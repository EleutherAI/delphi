from dataclasses import dataclass

import dspy

from ..config import ClientConfig
from ..logger import logger


@dataclass
class Response:
    text: str
    logprobs: list[float] | None = None
    prompt_logprobs: list[float] | None = None


class DspyClient:
    """
    A Wrapper over DSPy's LM, handling both running the model locally
    or using a remote API.
    Whether the model is running locally or remotely is chosen based on the
    input to the constructor.
    """

    provider = "dspy"

    def __init__(self, config: ClientConfig):
        self.config = config
        if self.config.provider == "local":
            import socket
            import subprocess
            import time
            from contextlib import closing

            import requests

            def find_free_port():
                with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
                    s.bind(("", 0))
                    s.listen(1)
                    port = s.getsockname()[1]
                return port

            # Find a free port for the VLLM server
            self.port = find_free_port()
            self.base_url = f"http://localhost:{self.port}"

            # Launch VLLM server via command line
            cmd = [
                "vllm",
                "serve",
                self.config.model,
                "--port",
                str(self.port),
                "--max-model-len",
                str(self.config.max_model_len),
                "--gpu-memory-utilization",
                str(self.config.gpu_memory_utilization),
                "--tensor-parallel-size",
                str(self.config.tensor_parallel_size),
                "--disable-log-requests",
            ]
            if self.config.enable_prefix_caching:
                cmd.append("--enable-prefix-caching")

            logger.info(f"Starting VLLM server with command: {' '.join(cmd)}")
            self.server_process = subprocess.Popen(cmd)

            # Wait for server to be ready
            max_wait_time = 300  # 5 minutes
            wait_interval = 5
            elapsed_time = 0

            while elapsed_time < max_wait_time:
                try:
                    response = requests.get(f"{self.base_url}/health", timeout=5)
                    if response.status_code == 200:
                        logger.info(f"VLLM server is ready at {self.base_url}")
                        break
                except requests.exceptions.RequestException:
                    pass

                time.sleep(wait_interval)
                elapsed_time += wait_interval
                logger.info(f"Waiting for VLLM server to start... ({elapsed_time}s)")
            else:
                raise RuntimeError(
                    f"VLLM server failed to start within {max_wait_time} seconds"
                )

            # Use DSPy with the local VLLM server
            self.client = dspy.LM(
                model=f"openai/{self.config.model}",
                api_base=f"{self.base_url}/v1",
                api_key="local",
                model_type="chat",
                max_tokens=self.config.number_tokens_to_generate,
            )
        elif self.config.provider == "openrouter":
            # Use openrouter API
            self.client = dspy.LM(
                model=f"openrouter/{self.config.model}", api_key=self.config.api_key
            )

    def close(self):
        if self.config.provider == "local":
            self.server_process.terminate()
            self.server_process.wait()
