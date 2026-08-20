"""Load retrieval configuration from YAML.

集中读取 retrieval.yaml 配置的地方。所有检索器（BM25 / Dense /
Hybrid）都应通过 load_config 获取参数，而不是各自解析 YAML。

之所以单独建这个模块，而不是把读 YAML 的逻辑塞进每个检索器：
1. 避免重复——三个检索器都要读同一份配置；
2. HF_ENDPOINT 这种全局副作用需要统一设置，散落各处容易漏；
3. 方便测试——可以传入临时配置文件路径。
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml

DEFAULT_CONFIG_PATH = Path("configs/retrieval.yaml")


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict:
    """Load retrieval config and apply HF mirror endpoint.

    Args:
        path: YAML 配置文件路径，默认为 configs/retrieval.yaml。

    Returns:
        解析后的配置字典，结构对应 retrieval.yaml 的顶层键
        （hf / bm25 / dense / ...）。

    Raises:
        FileNotFoundError: 配置文件不存在时抛出。

    Notes:
        该函数会读取 hf.endpoint 并写入 os.environ["HF_ENDPOINT"]。
        这一步必须在 sentence-transformers 加载模型**之前**完成，
        否则模型下载会尝试直连 huggingface.co 并超时。
        因此调用方应在 import 检索器之前就调用本函数。
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    # 国内无法直连 huggingface.co，必须走镜像（见 retrieval.yaml 的注释）。
    # 必须在任何 huggingface 库加载模型之前设置此环境变量。
    hf_endpoint = config.get("hf", {}).get("endpoint")
    if hf_endpoint:
        os.environ["HF_ENDPOINT"] = hf_endpoint

    return config
