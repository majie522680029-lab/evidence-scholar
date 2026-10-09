"""Launch the EvidenceScholar API server (C4).

启动：
    ES_LLM_MODE=fake python scripts/run_api_server.py
    ES_LLM_MODE=vllm python scripts/run_api_server.py

默认 fake 模式（不占卡），vllm 模式连 8000 端口的 vLLM。
端口默认 8001（避开 vLLM 的 8000）。
"""

from __future__ import annotations

import os

import uvicorn


def main() -> None:
    host = os.environ.get("ES_API_HOST", "0.0.0.0")
    port = int(os.environ.get("ES_API_PORT", "8001"))
    mode = os.environ.get("ES_LLM_MODE", "fake")
    print(f"Starting EvidenceScholar API on {host}:{port} (LLM mode: {mode})")
    uvicorn.run(
        "evidence_scholar.server.app:app",
        host=host,
        port=port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
