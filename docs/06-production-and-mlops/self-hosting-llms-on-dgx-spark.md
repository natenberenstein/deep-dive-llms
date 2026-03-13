# Self-Hosting LLMs on NVIDIA DGX Spark

> **TL;DR:** This is a practical, end-to-end guide for deploying embedding and chat completion models on NVIDIA DGX Spark systems using vLLM, with LiteLLM as a unified API gateway. It covers local testing on a single node, production deployment across multiple nodes, and client integration — all designed for air-gapped or restricted corporate networks where models must be cloned from ModelScope rather than pulled from Hugging Face.

## Table of Contents
- [Context and Constraints](#context-and-constraints)
- [Hardware Overview: DGX Spark](#hardware-overview-dgx-spark)
- [Prerequisites](#prerequisites)
- [Acquiring Models in a Restricted Network](#acquiring-models-in-a-restricted-network)
- [Part 1: Local Testing](#part-1-local-testing)
- [Part 2: Production Deployment](#part-2-production-deployment)
- [Part 3: Client Integration](#part-3-client-integration)
- [Monitoring and Observability](#monitoring-and-observability)
- [Troubleshooting](#troubleshooting)
- [Key Takeaways](#key-takeaways)
- [References](#references)

## Context and Constraints

This guide assumes the following environment:

| Constraint | Detail |
|---|---|
| **Hardware** | 2x NVIDIA DGX Spark systems |
| **Network** | Corporate/air-gapped — no access to Hugging Face Hub or ModelScope APIs |
| **Model source** | Git clone from ModelScope (e.g., `git clone https://www.modelscope.ai/...`) |
| **Serving framework** | vLLM (OpenAI-compatible API) |
| **API gateway** | LiteLLM (unified routing, API key management, usage tracking) |
| **Target models** | Qwen3-Embedding-8B (embeddings), Qwen3-8B or larger (chat completion) |

### Why This Setup?

```mermaid
graph LR
    A[Users] -->|OpenAI SDK| N["nginx<br/>llms.skhms.com"]
    N --> B[LiteLLM Gateway]
    B -->|"model: qwen3-embed"| C["DGX Spark #1<br/>vLLM — Embedding"]
    B -->|"model: qwen3-chat"| D["DGX Spark #2<br/>vLLM — Chat"]

    style N fill:#9b59b6,stroke:#8e44ad,color:#fff
    style B fill:#f39c12,stroke:#c87f0a,color:#fff
    style C fill:#3498db,stroke:#2980b9,color:#fff
    style D fill:#2ecc71,stroke:#27ae60,color:#fff
```

- **vLLM** provides high-throughput serving with PagedAttention, continuous batching, and an OpenAI-compatible API out of the box.
- **LiteLLM** sits in front as a unified gateway, giving users a single endpoint with clean model names, API key management, usage tracking, and rate limiting.
- **DGX Spark** offers 128 GB of unified memory per node — enough to run 8B models in FP16 comfortably, or quantized models up to ~70B.

## Hardware Overview: DGX Spark

Each DGX Spark system provides:

| Spec | Value |
|---|---|
| **Chip** | NVIDIA GB10 Grace Blackwell Superchip |
| **Architecture** | ARM-based (sm_121) |
| **CPU** | 20 cores (10x Cortex-X925 + 10x Cortex-A725) |
| **Memory** | 128 GB unified LPDDR5x (shared CPU/GPU) |
| **Memory bandwidth** | 273 GB/s |
| **AI compute** | Up to 1 PFLOP (FP4) |
| **Storage** | 4 TB SSD |
| **Power** | 240W |

**Key architectural note:** The DGX Spark uses a **unified memory architecture** — CPU and GPU share the same 128 GB address space. There is no separate VRAM. This eliminates CPU-to-GPU transfer overhead but means all system processes share the memory pool.

### What Fits on a DGX Spark?

| Model | Precision | Memory Required | Fits? |
|---|---|---|---|
| Qwen3-Embedding-8B | FP16 | ~16 GB | Yes — plenty of headroom |
| Qwen3-8B | FP16 | ~16 GB | Yes |
| Qwen3-32B | FP16 | ~64 GB | Yes, with limited KV cache |
| Qwen3-30B-A3B (MoE) | FP16 | ~60 GB | Yes — only 3B active params |
| Qwen3-72B | INT4 (GPTQ/AWQ) | ~36 GB | Yes, quantized |
| Qwen3-72B | FP16 | ~144 GB | No — exceeds 128 GB |

## Prerequisites

### 1. Git LFS

Model weights are stored as Git LFS objects. Git LFS must be installed and configured **before** cloning:

```bash
# Install Git LFS
sudo apt-get install git-lfs   # Debian/Ubuntu
# or
sudo dnf install git-lfs       # RHEL/Fedora

# Initialize (one-time)
git lfs install
```

> **Air-gapped note:** If `git lfs` is unavailable or blocked, you can download `.safetensors` weight files individually from the ModelScope web interface and place them in the model directory manually. The directory structure must match what the model's `config.json` expects.

### 2. Docker (Recommended)

NVIDIA provides a pre-built vLLM Docker image optimized for DGX Spark's sm_121 architecture:

```bash
# Verify Docker and NVIDIA Container Toolkit are installed
docker --version
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.8-base nvidia-smi
```

### 3. vLLM Compatibility on DGX Spark

**This is the most important prerequisite to understand.** The DGX Spark uses the sm_121 GPU architecture, which upstream vLLM does not fully support as of early 2026. You have two options:

| Option | Pros | Cons |
|---|---|---|
| **NVIDIA's pre-built vLLM container** | Works out of the box, tested for DGX Spark | May lag behind upstream vLLM features |
| **Build vLLM from source** | Latest features, full control | Requires patching Triton for sm_121, more complex setup |

The `--enforce-eager` flag is **required** on DGX Spark — it disables CUDA graph optimizations that aren't compatible with sm_121. This results in approximately 20-30% slower inference compared to CUDA graph mode on supported architectures, but it is the only reliable option.

## Acquiring Models in a Restricted Network

### Clone from ModelScope

```bash
# Create a central model store
sudo mkdir -p /models
sudo chown $USER:$USER /models

# Clone embedding model
git clone https://www.modelscope.ai/Qwen/Qwen3-Embedding-8B.git /models/Qwen3-Embedding-8B

# Clone chat model
git clone https://www.modelscope.ai/Qwen/Qwen3-8B.git /models/Qwen3-8B
```

### Verify Model Files

After cloning, confirm the weight files are actual tensors (not LFS pointers):

```bash
# Check file sizes — safetensors files should be multiple GB, not 130 bytes
ls -lh /models/Qwen3-Embedding-8B/*.safetensors

# If files are tiny (~130 bytes), LFS didn't pull the actual weights
# Fix with:
cd /models/Qwen3-Embedding-8B && git lfs pull
```

### Qwen3-Embedding-8B Model Details

| Property | Value |
|---|---|
| **Parameters** | 8 billion |
| **Max context length** | 32,768 tokens |
| **Embedding dimensions** | 32 to 4,096 (configurable, Matryoshka) |
| **Default dimension** | 4,096 |
| **FP16 memory** | ~16 GB |

The model uses Matryoshka Representation Learning, meaning you can truncate embeddings to smaller dimensions (e.g., 512, 1024) while preserving most semantic quality — useful for reducing storage and search costs in vector databases.

---

## Part 1: Local Testing

Use this section to validate your setup on a single DGX Spark before deploying to production.

### 1.1 Pull the vLLM Container

```bash
# NVIDIA's DGX Spark optimized image
docker pull nvcr.io/nvidia/vllm:dgx-spark-latest
```

> If pulling from a registry is not possible on your network, you can export the image on a machine with access (`docker save`) and import it on the DGX Spark (`docker load`).

### 1.2 Test the Embedding Model

```bash
docker run --rm --gpus all \
  -v /models/Qwen3-Embedding-8B:/model \
  -p 8000:8000 \
  nvcr.io/nvidia/vllm:dgx-spark-latest \
  vllm serve /model \
    --task embed \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len 8192 \
    --enforce-eager \
    --dtype float16
```

Key flags explained:

| Flag | Purpose |
|---|---|
| `--task embed` | Tells vLLM to serve as an embedding model (pooling runner) |
| `--enforce-eager` | **Required on DGX Spark** — disables incompatible CUDA graphs |
| `--max-model-len 8192` | Limits context window to save memory; increase up to 32768 if needed |
| `--dtype float16` | Use FP16 precision; change to `bfloat16` if supported |

Verify it's running:

```bash
# Check health
curl http://localhost:8000/health

# List models
curl http://localhost:8000/v1/models

# Generate an embedding
curl http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/model",
    "input": "This is a test sentence for embedding."
  }'
```

Expected: a JSON response with an `embedding` array of 4,096 floats.

### 1.3 Test the Chat Model

Stop the embedding container first (or use a different port), then:

```bash
docker run --rm --gpus all \
  -v /models/Qwen3-8B:/model \
  -p 8001:8001 \
  nvcr.io/nvidia/vllm:dgx-spark-latest \
  vllm serve /model \
    --host 0.0.0.0 \
    --port 8001 \
    --max-model-len 8192 \
    --enforce-eager \
    --dtype float16
```

Verify:

```bash
curl http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/model",
    "messages": [
      {"role": "user", "content": "Explain what vLLM is in two sentences."}
    ],
    "max_tokens": 128
  }'
```

### 1.4 Test Both on One Node (Optional)

An 8B embedding model (~16 GB) and an 8B chat model (~16 GB) can coexist on one DGX Spark (128 GB unified memory). This is useful for local testing but not recommended for production throughput:

```bash
# Terminal 1: Embedding model on port 8000
docker run --rm --gpus all \
  -v /models/Qwen3-Embedding-8B:/model \
  -p 8000:8000 \
  nvcr.io/nvidia/vllm:dgx-spark-latest \
  vllm serve /model \
    --task embed \
    --host 0.0.0.0 --port 8000 \
    --max-model-len 8192 \
    --enforce-eager --dtype float16 \
    --gpu-memory-utilization 0.4

# Terminal 2: Chat model on port 8001
docker run --rm --gpus all \
  -v /models/Qwen3-8B:/model \
  -p 8001:8001 \
  nvcr.io/nvidia/vllm:dgx-spark-latest \
  vllm serve /model \
    --host 0.0.0.0 --port 8001 \
    --max-model-len 8192 \
    --enforce-eager --dtype float16 \
    --gpu-memory-utilization 0.4
```

The `--gpu-memory-utilization 0.4` flag limits each instance to 40% of available GPU memory, preventing OOM when running side by side.

### 1.5 Quick Smoke Test Script

```python
"""smoke_test.py — Validate both endpoints are working."""
from openai import OpenAI

EMBED_URL = "http://localhost:8000/v1"
CHAT_URL = "http://localhost:8001/v1"

# Test embedding
embed_client = OpenAI(base_url=EMBED_URL, api_key="unused")
embed_resp = embed_client.embeddings.create(model="/model", input="test")
assert len(embed_resp.data[0].embedding) > 0
print(f"Embedding OK — dimension: {len(embed_resp.data[0].embedding)}")

# Test chat
chat_client = OpenAI(base_url=CHAT_URL, api_key="unused")
chat_resp = chat_client.chat.completions.create(
    model="/model",
    messages=[{"role": "user", "content": "Say hello in one word."}],
    max_tokens=16,
)
assert chat_resp.choices[0].message.content
print(f"Chat OK — response: {chat_resp.choices[0].message.content}")
```

```bash
pip install openai
python smoke_test.py
```

---

## Part 2: Production Deployment

### 2.1 Architecture

```mermaid
graph TD
    Users["Users<br/>(OpenAI SDK, curl, apps)"]
    Nginx["nginx Reverse Proxy<br/>llms.skhms.com:80/443"]
    LiteLLM["LiteLLM Proxy<br/>localhost:4000"]
    Spark1["DGX Spark #1<br/>vLLM — Embedding<br/>10.x.x.10:8000"]
    Spark2["DGX Spark #2<br/>vLLM — Chat<br/>10.x.x.11:8000"]
    Prometheus["Prometheus"]
    Grafana["Grafana Dashboard"]

    Users -->|"POST /v1/embeddings<br/>POST /v1/chat/completions"| Nginx
    Nginx --> LiteLLM
    LiteLLM -->|"model: qwen3-embed"| Spark1
    LiteLLM -->|"model: qwen3-chat"| Spark2
    Spark1 -->|"/metrics"| Prometheus
    Spark2 -->|"/metrics"| Prometheus
    Prometheus --> Grafana

    style Nginx fill:#9b59b6,stroke:#8e44ad,color:#fff
    style LiteLLM fill:#f39c12,stroke:#c87f0a,color:#fff
    style Spark1 fill:#3498db,stroke:#2980b9,color:#fff
    style Spark2 fill:#2ecc71,stroke:#27ae60,color:#fff
    style Prometheus fill:#e74c3c,stroke:#c0392b,color:#fff
```

**One model per node** — this maximizes available memory for KV cache and concurrent requests.

### 2.2 Deploy vLLM on DGX Spark #1 (Embedding)

Create a systemd service or use Docker Compose for persistence:

**Option A: Docker Compose (recommended)**

```yaml
# /opt/vllm/docker-compose.yml on DGX Spark #1
services:
  vllm-embedding:
    image: nvcr.io/nvidia/vllm:dgx-spark-latest
    runtime: nvidia
    ports:
      - "8000:8000"
    volumes:
      - /models/Qwen3-Embedding-8B:/model
    command: >
      vllm serve /model
        --task embed
        --host 0.0.0.0
        --port 8000
        --max-model-len 8192
        --enforce-eager
        --dtype float16
        --gpu-memory-utilization 0.9
    restart: unless-stopped
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

```bash
cd /opt/vllm && docker compose up -d
```

### 2.3 Deploy vLLM on DGX Spark #2 (Chat)

```yaml
# /opt/vllm/docker-compose.yml on DGX Spark #2
services:
  vllm-chat:
    image: nvcr.io/nvidia/vllm:dgx-spark-latest
    runtime: nvidia
    ports:
      - "8000:8000"
    volumes:
      - /models/Qwen3-8B:/model
    command: >
      vllm serve /model
        --host 0.0.0.0
        --port 8000
        --max-model-len 8192
        --enforce-eager
        --dtype float16
        --gpu-memory-utilization 0.9
    restart: unless-stopped
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

### 2.4 Deploy LiteLLM Gateway

LiteLLM runs on any machine on the network — it's lightweight (Python process, no GPU needed). It can run on one of the DGX Sparks or on a separate VM.

**Configuration (`/opt/litellm/config.yaml`):**

```yaml
model_list:
  - model_name: qwen3-embed
    litellm_params:
      model: openai/Qwen3-Embedding-8B
      api_base: http://10.x.x.10:8000/v1
      api_key: unused

  - model_name: qwen3-chat
    litellm_params:
      model: openai/Qwen3-8B
      api_base: http://10.x.x.11:8000/v1
      api_key: unused

general_settings:
  master_key: sk-change-this-to-a-secure-key
```

**Docker Compose (`/opt/litellm/docker-compose.yml`):**

```yaml
services:
  litellm:
    image: ghcr.io/berriai/litellm:main-latest
    ports:
      - "4000:4000"
    volumes:
      - ./config.yaml:/app/config.yaml
    command: >
      --config /app/config.yaml
      --host 0.0.0.0
      --port 4000
    restart: unless-stopped
```

```bash
cd /opt/litellm && docker compose up -d
```

### 2.5 API Key Management

LiteLLM supports creating per-user or per-team API keys:

```bash
# Create a key for a team (use llm-gateway.internal:4000 or llms.skhms.com)
curl http://llm-gateway.internal:4000/key/generate \
  -H "Authorization: Bearer sk-change-this-to-a-secure-key" \
  -H "Content-Type: application/json" \
  -d '{
    "team_id": "data-science",
    "max_budget": 100,
    "models": ["qwen3-embed", "qwen3-chat"]
  }'
```

This returns a key like `sk-abc123...` that users include in their requests. LiteLLM tracks usage per key.

### 2.6 Quick Access (No DNS Required)

Once LiteLLM is running, users can immediately access it by IP and port — no DNS, no reverse proxy, no IT tickets:

```
http://10.x.x.20:4000
```

Or set a hostname via `/etc/hosts` on each client machine:

```
# /etc/hosts on client machines
10.x.x.20  llm-gateway.internal
```

Then use `http://llm-gateway.internal:4000`. This is the fastest way to verify remote access works before investing in a custom domain.

### 2.7 Custom Domain Setup (e.g., `llms.skhms.com`)

To give users a clean URL like `http://llms.skhms.com` (no port numbers), you need two things: an internal DNS record and a reverse proxy. **No external requests or public DNS changes are required** — this is entirely within your corporate network.

#### Step 1: Internal DNS Record

Ask your network/IT team to create an internal DNS A record:

```
llms.skhms.com  →  A  →  10.x.x.20  (IP of the machine running LiteLLM)
```

This is configured on your company's internal DNS server (Active Directory, Bind, Infoblox, etc.). If your company uses split-horizon DNS for `skhms.com`, the record goes in the internal zone only.

**For testing before DNS is set up**, you can use `/etc/hosts` on client machines:

```
# /etc/hosts
10.x.x.20  llms.skhms.com
```

#### Step 2: Reverse Proxy (nginx)

LiteLLM listens on port 4000, but users expect port 80 (HTTP) or 443 (HTTPS). Install nginx on the same machine as LiteLLM:

```bash
sudo apt-get install nginx   # Debian/Ubuntu
```

**HTTP configuration:**

```nginx
# /etc/nginx/conf.d/llms.conf
server {
    listen 80;
    server_name llms.skhms.com;

    location / {
        proxy_pass http://127.0.0.1:4000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_buffering off;           # required for streaming responses
        proxy_read_timeout 300s;       # LLM responses can take time
        proxy_send_timeout 300s;
    }
}
```

**HTTPS configuration (if your company requires TLS):**

Request a TLS certificate from your internal CA, then:

```nginx
# /etc/nginx/conf.d/llms.conf
server {
    listen 443 ssl;
    server_name llms.skhms.com;

    ssl_certificate     /etc/ssl/certs/llms.skhms.com.crt;
    ssl_certificate_key /etc/ssl/private/llms.skhms.com.key;

    location / {
        proxy_pass http://127.0.0.1:4000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_buffering off;
        proxy_read_timeout 300s;
        proxy_send_timeout 300s;
    }
}

# Optional: redirect HTTP to HTTPS
server {
    listen 80;
    server_name llms.skhms.com;
    return 301 https://$host$request_uri;
}
```

```bash
sudo nginx -t && sudo systemctl reload nginx
```

#### What to ask your IT team

1. "Can you create an internal DNS A record for `llms.skhms.com` pointing to `10.x.x.20`?"
2. If HTTPS is required: "Can you issue an internal TLS certificate for `llms.skhms.com`?"

Users then connect to `http://llms.skhms.com` (or `https://`) with no port numbers.

---

## Part 3: Client Integration

All examples below use the custom domain. If you haven't set that up yet, swap the base URL:

| Setup | Base URL |
|---|---|
| **Quick access (no DNS)** | `http://10.x.x.20:4000/v1` or `http://llm-gateway.internal:4000/v1` |
| **Custom domain (with nginx)** | `http://llms.skhms.com/v1` |

### Python (OpenAI SDK)

```python
from openai import OpenAI

# Use one of:
#   "http://llm-gateway.internal:4000/v1"  (quick access, no DNS/nginx needed)
#   "http://llms.skhms.com/v1"             (custom domain with nginx)
BASE_URL = "http://llms.skhms.com/v1"

client = OpenAI(
    base_url=BASE_URL,
    api_key="sk-your-team-key",
)

# --- Embeddings ---
embed_response = client.embeddings.create(
    model="qwen3-embed",
    input=["First document to embed", "Second document to embed"],
)
for item in embed_response.data:
    print(f"Embedding dimension: {len(item.embedding)}")

# --- Chat Completion ---
chat_response = client.chat.completions.create(
    model="qwen3-chat",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is retrieval-augmented generation?"},
    ],
    max_tokens=256,
)
print(chat_response.choices[0].message.content)

# --- Streaming Chat ---
stream = client.chat.completions.create(
    model="qwen3-chat",
    messages=[{"role": "user", "content": "Explain transformers briefly."}],
    max_tokens=256,
    stream=True,
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

### curl

```bash
# Replace llms.skhms.com with llm-gateway.internal:4000 if using quick access

# Embedding
curl http://llms.skhms.com/v1/embeddings \
  -H "Authorization: Bearer sk-your-team-key" \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen3-embed", "input": "Hello world"}'

# Chat completion
curl http://llms.skhms.com/v1/chat/completions \
  -H "Authorization: Bearer sk-your-team-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-chat",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 64
  }'
```

### JavaScript / TypeScript

```typescript
import OpenAI from "openai";

const client = new OpenAI({
  baseURL: "http://llms.skhms.com/v1",
  apiKey: "sk-your-team-key",
});

// Embedding
const embedding = await client.embeddings.create({
  model: "qwen3-embed",
  input: "Text to embed",
});
console.log(embedding.data[0].embedding.length);

// Chat
const chat = await client.chat.completions.create({
  model: "qwen3-chat",
  messages: [{ role: "user", content: "Hello" }],
});
console.log(chat.choices[0].message.content);
```

### LangChain Integration

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

embeddings = OpenAIEmbeddings(
    model="qwen3-embed",
    openai_api_base="http://llms.skhms.com/v1",
    openai_api_key="sk-your-team-key",
)

llm = ChatOpenAI(
    model="qwen3-chat",
    openai_api_base="http://llms.skhms.com/v1",
    openai_api_key="sk-your-team-key",
)

# These can now be used in any LangChain pipeline (RAG, agents, etc.)
```

---

## Monitoring and Observability

### vLLM Prometheus Metrics

vLLM exposes Prometheus-format metrics at `/metrics` on each node:

```bash
curl http://10.x.x.10:8000/metrics
curl http://10.x.x.11:8000/metrics
```

Key metrics to watch:

| Metric | What It Tells You |
|---|---|
| `vllm:num_requests_running` | Current concurrent requests |
| `vllm:num_requests_waiting` | Requests queued (indicates saturation) |
| `vllm:gpu_cache_usage_perc` | KV cache utilization — if consistently >90%, reduce `max-model-len` or increase memory |
| `vllm:avg_generation_throughput_toks_per_s` | Tokens per second throughput |
| `vllm:e2e_request_latency_seconds` | End-to-end request latency |

### LiteLLM Usage Tracking

LiteLLM logs per-request usage. You can query it:

```bash
# Get usage for a specific key
curl http://llms.skhms.com/key/info \
  -H "Authorization: Bearer sk-change-this-to-a-secure-key" \
  -d '{"key": "sk-your-team-key"}'
```

### Grafana Dashboard (Optional)

Add a Prometheus scrape config for both vLLM instances:

```yaml
# prometheus.yml
scrape_configs:
  - job_name: vllm-embedding
    static_configs:
      - targets: ["10.x.x.10:8000"]
  - job_name: vllm-chat
    static_configs:
      - targets: ["10.x.x.11:8000"]
```

---

## Troubleshooting

### Model files are 130 bytes (LFS pointers, not actual weights)

```bash
cd /models/Qwen3-Embedding-8B
git lfs pull
```

If `git lfs` fails behind a proxy, download `.safetensors` files manually from the ModelScope web UI.

### vLLM crashes with CUDA error on DGX Spark

Ensure you're using:
1. NVIDIA's DGX Spark container (not upstream vLLM)
2. The `--enforce-eager` flag

```bash
# Verify GPU architecture
nvidia-smi --query-gpu=compute_cap --format=csv
# Should show sm_121 or similar
```

### Out of memory

- Lower `--gpu-memory-utilization` (e.g., 0.8)
- Lower `--max-model-len` (e.g., 4096)
- Use a quantized model variant (INT4/INT8)
- Don't run other GPU workloads alongside vLLM — the 128 GB is unified and shared

### LiteLLM can't reach vLLM backends

```bash
# From the LiteLLM host, verify connectivity
curl http://10.x.x.10:8000/health
curl http://10.x.x.11:8000/health
```

Check that Docker isn't binding to `127.0.0.1` — vLLM must listen on `0.0.0.0`.

### Slow inference (~20-30% slower than expected)

This is expected on DGX Spark due to `--enforce-eager`. The sm_121 architecture doesn't yet have full CUDA graph support in vLLM/Triton. Track NVIDIA's container releases for improvements.

---

## Key Takeaways

1. **DGX Spark requires NVIDIA's vLLM container** — upstream vLLM doesn't fully support the sm_121 architecture. Always use `--enforce-eager`.

2. **128 GB unified memory is generous but shared** — an 8B model in FP16 uses ~16 GB, leaving plenty of room. But don't run other heavy workloads alongside vLLM.

3. **One model per node for production** — dedicating each DGX Spark to a single model maximizes KV cache memory and concurrent request throughput.

4. **LiteLLM is the right abstraction layer** — it turns two separate vLLM instances into a single, manageable API gateway with clean model names, API keys, and usage tracking.

5. **Git LFS is the most common gotcha** — always verify that cloned model files are actual weights (multiple GB), not 130-byte LFS pointers.

6. **Any OpenAI SDK client works** — because both vLLM and LiteLLM expose OpenAI-compatible APIs, users can integrate with Python, TypeScript, curl, LangChain, or any tool that supports the OpenAI API format.

7. **Start with local testing, then go to production** — validate on a single node with the smoke test script before deploying Docker Compose services and the LiteLLM gateway.

## References

### Hardware
1. [NVIDIA DGX Spark Product Page](https://www.nvidia.com/en-us/products/workstations/dgx-spark/) — Official specs and overview
2. [DGX Spark Hardware Overview](https://docs.nvidia.com/dgx/dgx-spark/hardware.html) — Detailed hardware documentation
3. [NVIDIA DGX Spark In-Depth Review (LMSYS)](https://lmsys.org/blog/2025-10-13-nvidia-dgx-spark/) — Performance benchmarks and analysis

### vLLM on DGX Spark
4. [vLLM for DGX Spark (NVIDIA)](https://build.nvidia.com/spark/vllm) — NVIDIA's optimized vLLM container
5. [vLLM DGX Spark Compatibility Discussion](https://discuss.vllm.ai/t/nvidia-dgx-spark-compatibility/1756) — Community discussion on sm_121 support
6. [vLLM Installation on DGX Spark Guide](https://medium.com/@stablehigashi/vllm-installation-on-dgx-spark-gb10-sm-121-and-qwen-3-5-serving-guide-9eba91e448f8) — Step-by-step source build guide
7. [Building vLLM from Source on DGX Spark](https://medium.com/@anveshkumarchavidi/installing-vllm-on-nvidia-dgx-spark-from-source-4dde137ff3ef) — Alternative source build approach
8. [dgx-spark-vllm-setup (GitHub)](https://github.com/eelbaz/dgx-spark-vllm-setup) — One-command setup script

### vLLM Documentation
9. [vLLM Serve CLI Reference](https://docs.vllm.ai/en/stable/cli/serve/) — Command-line options
10. [vLLM Embedding Documentation](https://docs.vllm.ai/en/v0.7.0/getting_started/examples/embedding.html) — Embedding model serving guide
11. [vLLM OpenAI-Compatible Server](https://docs.vllm.ai/en/stable/serving/openai_compatible_server/) — API compatibility details

### Models
12. [Qwen3-Embedding-8B (Hugging Face)](https://huggingface.co/Qwen/Qwen3-Embedding-8B) — Model card and specifications
13. [Qwen3-Embedding (GitHub)](https://github.com/QwenLM/Qwen3-Embedding) — Source repository and documentation

### LiteLLM
14. [LiteLLM Documentation](https://docs.litellm.ai/) — Proxy configuration, API key management, and usage tracking
15. [LiteLLM GitHub](https://github.com/BerriAI/litellm) — Source code and Docker images
