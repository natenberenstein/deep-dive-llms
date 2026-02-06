# Tools & Frameworks

> A reference guide to the LLM ecosystem — providers, frameworks, vector databases, and evaluation tools. Organized by category with guidance on when to use each.

## Table of Contents
- [LLM Providers](#llm-providers)
- [Agent & Application Frameworks](#agent--application-frameworks)
- [Vector Databases](#vector-databases)
- [Embedding Models](#embedding-models)
- [Evaluation & Observability](#evaluation--observability)
- [Data & Document Processing](#data--document-processing)

---

## LLM Providers

| Name | What It Does | When to Use It | Link |
|---|---|---|---|
| **OpenAI** | GPT-4o, GPT-4, o1/o3 reasoning models. Leading commercial API. | Default choice for most applications; strong function calling and instruction following | [platform.openai.com](https://platform.openai.com) |
| **Anthropic** | Claude 4 family. Strong at long context, safety, and complex reasoning. | When you need long-context reliability, nuanced reasoning, or safety-critical applications | [anthropic.com](https://www.anthropic.com) |
| **Google** | Gemini models. Multimodal (text, image, video, audio). | Multimodal applications; tight integration with Google Cloud | [ai.google.dev](https://ai.google.dev) |
| **Meta (LLaMA)** | Open-weight LLaMA models (8B–405B). Run locally or on your infra. | When you need full control, data privacy, or to avoid API costs at scale | [llama.meta.com](https://llama.meta.com) |
| **Mistral AI** | Open and commercial models. Strong efficiency at smaller sizes. | Cost-efficient deployments; European data sovereignty requirements | [mistral.ai](https://mistral.ai) |
| **DeepSeek** | Open models with strong reasoning capabilities (DeepSeek-R1). | Reasoning-heavy tasks; open-source alternative to o1 | [deepseek.com](https://www.deepseek.com) |
| **Ollama** | Local LLM runner. Download and run open models with one command. | Local development, prototyping, privacy-sensitive workloads | [ollama.com](https://ollama.com) |
| **vLLM** | High-performance LLM serving engine. | Production self-hosted deployments requiring maximum throughput | [github.com/vllm-project/vllm](https://github.com/vllm-project/vllm) |

## Agent & Application Frameworks

| Name | What It Does | When to Use It | Link |
|---|---|---|---|
| **LangChain** | Composable building blocks for LLM apps: chains, tools, agents, retrievers. | Rapid prototyping; broad ecosystem integrations; RAG pipelines | [langchain.com](https://www.langchain.com) |
| **LangGraph** | Stateful graph orchestration for agents. Cycles, conditional edges, persistence. | Production agent systems; multi-agent architectures; complex control flow | [langchain-ai.github.io/langgraph](https://langchain-ai.github.io/langgraph/) |
| **LangSmith** | Observability and evaluation platform for LangChain/LangGraph apps. | Debugging, tracing, and evaluating LLM applications in development and production | [smith.langchain.com](https://smith.langchain.com) |
| **LlamaIndex** | Data framework for connecting LLMs with external data. Specialized for RAG. | Data-heavy RAG pipelines; complex document ingestion and querying | [llamaindex.ai](https://www.llamaindex.ai) |
| **CrewAI** | Multi-agent framework with role-based agents. | Quick multi-agent prototypes; role-based task delegation | [crewai.com](https://www.crewai.com) |
| **AutoGen** | Multi-agent conversation framework from Microsoft. | Multi-agent research; conversational agent systems; human-in-the-loop | [github.com/microsoft/autogen](https://github.com/microsoft/autogen) |
| **Haystack** | End-to-end NLP/LLM framework by deepset. Pipeline-based architecture. | Production search and RAG pipelines; strong on document processing | [haystack.deepset.ai](https://haystack.deepset.ai) |
| **Semantic Kernel** | Microsoft's SDK for integrating LLMs into applications. | Enterprise .NET/Java/Python environments; Azure-centric stacks | [github.com/microsoft/semantic-kernel](https://github.com/microsoft/semantic-kernel) |

## Vector Databases

See the [detailed comparison](../docs/02-retrieval-augmented-generation/vector-databases.md) for in-depth analysis.

| Name | What It Does | When to Use It | Link |
|---|---|---|---|
| **Qdrant** | Purpose-built vector DB in Rust. Fast, simple, great filtering. | Best default for new projects; under 100M vectors | [qdrant.tech](https://qdrant.tech) |
| **Milvus / Zilliz** | Distributed vector DB for massive scale. Multiple index types, GPU support. | Billion-scale deployments; need GPU acceleration or diverse index types | [milvus.io](https://milvus.io) |
| **Elasticsearch** | Text search engine with vector capabilities (kNN, HNSW). | Existing ES users; hybrid keyword + vector search | [elastic.co](https://www.elastic.co) |
| **Pinecone** | Fully managed vector DB. Zero ops. | Teams wanting zero operational overhead; quick start | [pinecone.io](https://www.pinecone.io) |
| **Weaviate** | Vector DB with built-in vectorization modules. | When you want the DB to handle embedding generation | [weaviate.io](https://weaviate.io) |
| **ChromaDB** | Lightweight, developer-friendly vector DB. Embeds in Python. | Prototyping; small-scale applications; local development | [trychroma.com](https://www.trychroma.com) |
| **pgvector** | Vector extension for PostgreSQL. | Teams already on PostgreSQL who want vector search without new infrastructure | [github.com/pgvector/pgvector](https://github.com/pgvector/pgvector) |

## Embedding Models

| Name | What It Does | When to Use It | Link |
|---|---|---|---|
| **OpenAI text-embedding-3** | Commercial embedding models (small: 512d, large: 3072d). | Default choice; strong general-purpose performance | [platform.openai.com](https://platform.openai.com/docs/guides/embeddings) |
| **Cohere Embed v3** | Multilingual embeddings with compression support. | Multilingual applications; cost-efficient with int8/binary quantization | [cohere.com](https://cohere.com) |
| **Voyage AI** | Specialized embeddings for code, legal, finance. | Domain-specific retrieval where general embeddings underperform | [voyageai.com](https://www.voyageai.com) |
| **BGE / GTE** | Open-source embeddings (BAAI, Alibaba). Run locally. | Self-hosted deployments; privacy requirements; cost optimization | [huggingface.co](https://huggingface.co/BAAI/bge-large-en-v1.5) |
| **Sentence Transformers** | Python library for computing embeddings with any compatible model. | Local embedding computation; fine-tuning embeddings on your domain | [sbert.net](https://www.sbert.net) |

## Evaluation & Observability

| Name | What It Does | When to Use It | Link |
|---|---|---|---|
| **LangSmith** | Tracing, evaluation, and monitoring for LLM apps. | LangChain/LangGraph projects; debugging agent behavior | [smith.langchain.com](https://smith.langchain.com) |
| **Weights & Biases (W&B)** | Experiment tracking, model monitoring, prompt management. | Training and fine-tuning workflows; experiment comparison | [wandb.ai](https://wandb.ai) |
| **Ragas** | RAG evaluation framework. Measures faithfulness, relevancy, context recall. | Evaluating RAG pipeline quality with automated metrics | [ragas.io](https://ragas.io) |
| **Phoenix (Arize)** | LLM observability: traces, evaluations, datasets. | Production monitoring; debugging retrieval and generation quality | [phoenix.arize.com](https://phoenix.arize.com) |
| **Braintrust** | Evaluation, logging, and prompt playground for LLM apps. | Prompt iteration and A/B testing; scoring and comparison | [braintrust.dev](https://www.braintrust.dev) |
| **promptfoo** | CLI and library for testing and evaluating prompts. | CI/CD prompt testing; comparing models and prompts systematically | [promptfoo.dev](https://www.promptfoo.dev) |

## Data & Document Processing

| Name | What It Does | When to Use It | Link |
|---|---|---|---|
| **Unstructured** | Ingestion library for parsing PDFs, DOCXs, HTML, images, and more. | Document ingestion in RAG pipelines; handles messy real-world files | [unstructured.io](https://unstructured.io) |
| **LangChain Document Loaders** | 100+ loaders for PDFs, web pages, databases, APIs, etc. | Quick integration of diverse data sources into LangChain pipelines | [docs.langchain.com](https://docs.langchain.com) |
| **Docling** | IBM's document conversion library. PDF, DOCX, HTML to structured output. | High-fidelity document conversion with layout understanding | [github.com/DS4SD/docling](https://github.com/DS4SD/docling) |
| **Marker** | Converts PDFs to Markdown with high accuracy. | PDF-to-text conversion for RAG; preserves structure and formatting | [github.com/VikParuchuri/marker](https://github.com/VikParuchuri/marker) |
