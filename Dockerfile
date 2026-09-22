# Hugging Face Spaces 部署镜像
# 只跑 FastAPI + Chainlit（main.py，监听 7860）；Streamlit Dashboard 由 Streamlit Cloud 单独托管。
# 刻意不含 Playwright / torch / sentence-transformers：云端不执行抓取与 Reranker。

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HOME=/home/user \
    PORT=7860

WORKDIR /app

# 依赖全部是 manylinux 预编译 wheel（chromadb / jieba / numpy / pandas 都不需要现场编译），
# 所以不装 gcc 工具链，只留 ca-certificates 和健康检查用的 curl。
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

# 先装依赖（单独一层）：之后只改代码时仍能命中缓存
COPY requirements-hf.txt /app/requirements-hf.txt
RUN pip install --no-cache-dir -r /app/requirements-hf.txt

# HF Spaces 推荐非 root 运行
RUN useradd -m -u 1000 user

COPY --chown=user:user . /app

# 运行期要写数据库 / 日志，先建好目录并交还属主
RUN mkdir -p /app/agent/data /app/logs \
    && chown -R user:user /app

USER user

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -fsS http://127.0.0.1:7860/docs || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
