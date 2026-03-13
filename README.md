# Deep Dive into LLMs & AI Agents

> A structured, tutorial-style knowledge base covering Large Language Models, Retrieval-Augmented Generation, and AI Agents — from foundations to advanced architectures.

[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

## Learning Roadmap

```mermaid
graph LR
    A[LLM Foundations] --> B[Hardware & Infrastructure]
    A --> C[Retrieval-Augmented Generation]
    C --> D[AI Agents]

    A --> A1[How LLMs Work]
    A --> A2[Training & Fine-Tuning]

    B --> B1[GPU Architecture]
    B --> B2[Infrastructure & Scaling]
    B --> B3[Energy Efficiency]

    C --> C1[Chunking Strategies]
    C --> C2[Context Engineering]
    C --> C3[Vector Databases]

    D --> D1[Agent Fundamentals]
    D --> D2[Multi-Agent Architectures]

    style A fill:#4a90d9,stroke:#2c5f8a,color:#fff
    style B fill:#f39c12,stroke:#c87f0a,color:#fff
    style C fill:#7b68ee,stroke:#5a4cb5,color:#fff
    style D fill:#e06c75,stroke:#a84e55,color:#fff
    style A1 fill:#6aacf0,stroke:#4a8ad0,color:#fff
    style A2 fill:#6aacf0,stroke:#4a8ad0,color:#fff
    style B1 fill:#f5a962,stroke:#d68e4a,color:#fff
    style B2 fill:#f5a962,stroke:#d68e4a,color:#fff
    style B3 fill:#f5a962,stroke:#d68e4a,color:#fff
    style C1 fill:#9a8aee,stroke:#7a6ace,color:#fff
    style C2 fill:#9a8aee,stroke:#7a6ace,color:#fff
    style C3 fill:#9a8aee,stroke:#7a6ace,color:#fff
    style D1 fill:#e89a9f,stroke:#c87a7f,color:#fff
    style D2 fill:#e89a9f,stroke:#c87a7f,color:#fff
```

## Table of Contents

### 1. LLM Foundations
- [Section Overview](docs/01-llm-foundations/README.md)
- [How LLMs Work](docs/01-llm-foundations/how-llms-work.md) — Evolution from statistical models to modern LLMs, Transformer architecture, emergent abilities
- [Training & Fine-Tuning](docs/01-llm-foundations/training-and-fine-tuning.md) — Pre-training, SFT, RLHF, DPO, and practical considerations

### 2. Hardware & Infrastructure
- [Section Overview](docs/04-hardware-and-infrastructure/README.md)
- [GPU Architecture for LLMs](docs/04-hardware-and-infrastructure/gpu-architecture-for-llms.md) — Why GPUs excel at transformers, tensor cores, memory hierarchy, and GPU evolution (Volta → Ampere → Hopper)
- [Infrastructure & Scaling](docs/04-hardware-and-infrastructure/infrastructure-and-scaling.md) — Scaling laws, data center requirements, distributed training challenges, cost analysis, and the infrastructure arms race
- [Energy Efficiency](docs/04-hardware-and-infrastructure/energy-efficiency.md) — Power consumption, environmental impact, hardware and algorithmic optimizations, sustainability approaches

### 3. Retrieval-Augmented Generation
- [Section Overview](docs/02-retrieval-augmented-generation/README.md)
- [Chunking Strategies](docs/02-retrieval-augmented-generation/chunking-strategies.md) — How to split documents for optimal retrieval, based on Chroma research
- [Context Engineering](docs/02-retrieval-augmented-generation/context-engineering.md) — Why context degrades and how to fight it, based on Chroma research
- [Vector Databases](docs/02-retrieval-augmented-generation/vector-databases.md) — Elasticsearch vs. Qdrant vs. Milvus comparison

### 4. AI Agents
- [Section Overview](docs/03-ai-agents/README.md)
- [Agent Fundamentals](docs/03-ai-agents/agent-fundamentals.md) — ReAct, tool use, memory, planning
- [Memory and State Management](docs/03-ai-agents/memory-and-state-management.md) — Conversation memory, long-term recall, episodic memory, state machines, checkpointing
- [Multi-Agent Architectures](docs/03-ai-agents/multi-agent-architectures.md) — Patterns, LangChain, LangGraph

### Resources
- [Key Papers](resources/papers.md) — Curated reading list with annotations
- [Tools & Frameworks](resources/tools-and-frameworks.md) — Reference guide to the ecosystem
- [Hardware for LLMs](resources/hardware-for-llms.md) — GPUs, TPUs, memory, cost analysis, optimization

## Who Is This For?

- **ML/AI Engineers** who want a structured reference on LLM internals, RAG pipelines, and agent architectures
- **Software Engineers** transitioning into AI/ML who need practical, concept-first explanations
- **Technical Leaders** evaluating where LLMs and agents fit into their stack
- **Students & Researchers** looking for a curated, opinionated guide through the literature

## Quick-Start Reading Guide

| Your Goal | Start Here |
|---|---|
| Understand how LLMs work from scratch | [How LLMs Work](docs/01-llm-foundations/how-llms-work.md) |
| Understand GPU architecture and scaling | [GPU Architecture for LLMs](docs/04-hardware-and-infrastructure/gpu-architecture-for-llms.md) |
| Evaluate infrastructure requirements | [Infrastructure & Scaling](docs/04-hardware-and-infrastructure/infrastructure-and-scaling.md) |
| Optimize energy efficiency | [Energy Efficiency](docs/04-hardware-and-infrastructure/energy-efficiency.md) |
| Build or improve a RAG pipeline | [Chunking Strategies](docs/02-retrieval-augmented-generation/chunking-strategies.md) |
| Choose a vector database | [Vector Databases](docs/02-retrieval-augmented-generation/vector-databases.md) |
| Understand long context and when to use it vs. RAG | [Long Context and Context Windows](docs/01-llm-foundations/long-context-and-context-windows.md) |
| Understand AI agents | [Agent Fundamentals](docs/03-ai-agents/agent-fundamentals.md) |
| Design agent memory systems | [Memory and State Management](docs/03-ai-agents/memory-and-state-management.md) |
| Design multi-agent systems | [Multi-Agent Architectures](docs/03-ai-agents/multi-agent-architectures.md) |
| Find key papers to read | [Papers](resources/papers.md) |
| Choose hardware for training/inference | [Hardware for LLMs](resources/hardware-for-llms.md) |

## Contributing

Want to add a topic or improve existing content? See the [Contributing Guide](CONTRIBUTING.md) for templates, style conventions, and how to submit changes.
