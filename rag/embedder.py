"""本地 Embedding：fastembed（BAAI/bge-small-zh-v1.5）。

- 不再依赖任何云端 Embedding API：无需 Key，模型首次使用时自动下载并缓存到
  本地，之后可完全离线运行（本地无网时可用 HF_HUB_OFFLINE=1）。
- 输出维度 512。旧方案（火山方舟 doubao-embedding-vision）是 2048 维，
  换模型后必须删除 chroma_db 重新建库，否则 Chroma 会因维度不一致报错。
- 缓存目录默认 ~/.cache/fastembed（fastembed 自身默认放在系统临时目录，
  每次启动都会重新下载，所以这里显式固定到一个持久目录），
  可用环境变量 FASTEMBED_CACHE_PATH 覆盖。
- Windows 提醒：缓存默认用软链接，未开「开发者模式」时会报 WinError 1314，
  这里自动改成复制模式（HF_HUB_DISABLE_SYMLINKS=1）。
"""
import os
from pathlib import Path

from shared.config import LOCAL_EMBEDDING_MODEL

# bge-small-zh-v1.5 的向量维度
EMBEDDING_DIM = 512

# bge 中文系列建议只在「查询侧」加指令前缀，文档侧保持原样
QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："

# 模型缓存目录：必须在 import fastembed（内部 import huggingface_hub）之前设置
os.environ.setdefault(
    "FASTEMBED_CACHE_PATH",
    os.getenv("FASTEMBED_CACHE_PATH") or str(Path.home() / ".cache" / "fastembed"),
)
if os.name == "nt":
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")

_model = None


def _get_model():
    """懒加载本地模型：第一次调用才下载/载入，避免 import 就吃内存。

    把 import 也放进函数里，这样没装 fastembed 时 import rag.embedder
    仍然可用（真要用 embedding 时才报错）。
    """
    global _model
    if _model is None:
        from fastembed import TextEmbedding
        _model = TextEmbedding(model_name=LOCAL_EMBEDDING_MODEL)
    return _model


def _to_float_list(vec) -> list[float]:
    return [float(x) for x in vec]


def embed_one(text: str) -> list[float]:
    model = _get_model()
    return _to_float_list(next(iter(model.embed([text]))))


def embed_texts(texts: list[str], batch_size: int = 32) -> list[list[float]]:
    """批量生成文档向量（本地推理，按批推进并打印进度）。"""
    model = _get_model()
    total = len(texts)
    vectors = []
    for i, vec in enumerate(model.embed(texts, batch_size=batch_size), 1):
        vectors.append(_to_float_list(vec))
        if i % batch_size == 0 or i == total:
            print(f"  [{i}/{total}] 已生成向量（维度 {len(vectors[-1])}）")
    return vectors


def embed_query(query: str) -> list[float]:
    """查询向量：带上 bge 推荐的检索指令前缀。

    个别 fastembed 版本/模型不支持 query_embed，则退回普通向量，
    只是召回质量略降，不影响正确性。
    """
    model = _get_model()
    try:
        return _to_float_list(next(iter(model.query_embed([QUERY_INSTRUCTION + query]))))
    except Exception:                                # noqa: BLE001 - 不支持就退回普通编码
        return embed_one(query)


if __name__ == "__main__":
    texts = ["熟悉 Python", "会踢足球", "掌握 RAG 技术"]
    vecs = embed_texts(texts)
    print(f"\n共 {len(vecs)} 条，每条向量维度：{len(vecs[0])}")
