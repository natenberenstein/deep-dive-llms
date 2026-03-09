# Retrieval-Augmented Generation (RAG)

> Grounding LLM outputs in external knowledge — how to retrieve the right context and present it effectively.

## What This Section Covers

Retrieval-Augmented Generation (RAG) is the most common pattern for building LLM applications that need access to external or up-to-date knowledge. Instead of relying solely on what the model memorized during training, RAG retrieves relevant documents and includes them in the prompt.

This section covers the full RAG pipeline — from how you split documents, to how you store and search them, to how the model processes the retrieved context.

## The RAG Pipeline

```mermaid
graph LR
    subgraph Indexing
        A[Documents] --> B[Chunking]
        B --> C[Embedding]
        C --> D[Vector Database]
    end

    subgraph Retrieval
        E[User Query] --> F[Query Embedding]
        F --> G[Similarity Search]
        D --> G
        G --> H[Top-K Chunks]
    end

    subgraph Generation
        H --> I[Prompt Construction]
        E --> I
        I --> J[LLM]
        J --> K[Response]
    end

    style A fill:#f0f0f0,stroke:#ccc,color:#333
    style D fill:#7b68ee,stroke:#5a4cb5,color:#fff
    style J fill:#4a90d9,stroke:#2c5f8a,color:#fff
    style K fill:#90ee90,stroke:#6ecc6e,color:#333
```

Each stage of this pipeline introduces decisions that affect the final answer quality:

| Stage | Key Question | Covered In |
|---|---|---|
| **Chunking** | How do you split documents for optimal retrieval? | [Chunking Strategies](chunking-strategies.md) |
| **Embedding** | Which embedding model should you use? | [Embedding Models](embedding-models.md) |
| **Context Quality** | How does retrieved context degrade and how do you prevent it? | [Context Engineering](context-engineering.md) |
| **Storage & Search** | Which vector database should you use? | [Vector Databases](vector-databases.md) |
| **Advanced Retrieval** | How do you go beyond naive vector search? | [Advanced Retrieval Patterns](advanced-retrieval-patterns.md) |
| **Evaluation** | How do you measure RAG quality? | [RAG Evaluation](rag-evaluation.md) |

## Pages in This Section

| Page | What You'll Learn |
|---|---|
| [Chunking Strategies](chunking-strategies.md) | Evaluation of 6 chunking methods, optimal chunk sizes, and why defaults are often wrong |
| [Embedding Models](embedding-models.md) | Key embedding models compared, MTEB leaderboard, open vs proprietary, and fine-tuning |
| [Context Engineering](context-engineering.md) | Why LLM performance degrades with more context, and strategies to fight it |
| [Vector Databases](vector-databases.md) | How vector search works, and when to use Elasticsearch, Qdrant, or Milvus |
| [Advanced Retrieval Patterns](advanced-retrieval-patterns.md) | Hybrid search, re-ranking, HyDE, query transformation, and multi-hop retrieval |
| [RAG Evaluation](rag-evaluation.md) | RAGAS framework, component-level evaluation, failure modes, and evaluation tools |

## Suggested Reading Order

1. Start with **Chunking Strategies** — the first decision in any RAG pipeline
2. Then read **Embedding Models** — choosing the right model to convert text to vectors
3. Then read **Context Engineering** — understanding how context quality affects generation
4. Then read **Vector Databases** — choosing the right storage and retrieval infrastructure
5. Then read **Advanced Retrieval Patterns** — going beyond naive vector search
6. Finally, **RAG Evaluation** — measuring and improving your RAG pipeline
