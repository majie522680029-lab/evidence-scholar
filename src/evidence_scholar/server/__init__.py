"""Server package: FastAPI app exposing the agent (C4).

不在 __init__ 里重导出 app 实例，避免子模块名 app 被遮蔽成 FastAPI 对象，
导致 `import evidence_scholar.server.app as appmod` 拿到实例而非模块。
要用 app 实例时直接 `from evidence_scholar.server.app import app`。
"""
