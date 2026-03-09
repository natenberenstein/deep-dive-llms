# Learning Paths

This guide provides suggested reading orders through the deep-dive-llms
documentation, tailored for different roles and goals. Each path sequences
the documents so that concepts build on each other logically. Not every
reader needs to cover every document; these paths help you focus your time
on the material most relevant to your work.

Reading times are estimated at roughly 200 words per minute. Actual time
may vary depending on your familiarity with the subject matter and how
deeply you engage with the technical details.

---

## ML/AI Engineer

A comprehensive, depth-first path through all material. This path covers
theoretical foundations, implementation details, systems-level infrastructure,
and advanced topics like multi-agent architectures and energy efficiency. It
is designed for engineers who will be building, training, fine-tuning, or
deploying LLM-based systems and need to understand the full stack from
attention mechanisms to GPU memory management.

**Why this order:** The path starts with the model itself (architecture,
training), then moves to the hardware and infrastructure that constrains
what is practical, then to the retrieval and agent patterns that compose
models into applications.

1. [How LLMs Work](../docs/01-llm-foundations/how-llms-work.md)
   Core architecture: tokenization, embeddings, self-attention, Transformer
   blocks, positional encodings, and decoding strategies. This is the
   foundation everything else builds on.
   (~7 min)

2. [Training and Fine-Tuning](../docs/01-llm-foundations/training-and-fine-tuning.md)
   Pre-training objectives, scaling laws, the three-stage alignment pipeline
   (SFT, reward modeling, RLHF/DPO), and parameter-efficient methods like
   LoRA and QLoRA.
   (~8 min)

3. [GPU Architecture for LLMs](../docs/04-hardware-and-infrastructure/gpu-architecture-for-llms.md)
   GPU memory hierarchy (SRAM, HBM, DRAM), compute vs. memory boundedness,
   Flash Attention, KV cache sizing, quantization techniques, and arithmetic
   intensity analysis.
   (~12 min)

4. [Infrastructure and Scaling](../docs/04-hardware-and-infrastructure/infrastructure-and-scaling.md)
   Serving frameworks (vLLM, TGI, TensorRT-LLM), parallelism strategies
   (tensor, pipeline, data, expert), continuous batching, PagedAttention,
   speculative decoding, and deployment patterns.
   (~15 min)

5. [Vector Databases](../docs/02-retrieval-augmented-generation/vector-databases.md)
   Embedding models, ANN indexing algorithms (HNSW, IVF), distance metrics,
   and a comparison of production vector database options.
   (~7 min)

6. [Chunking Strategies](../docs/02-retrieval-augmented-generation/chunking-strategies.md)
   Document splitting methods (fixed-size, recursive, semantic), chunk sizing
   trade-offs, overlap parameters, and metadata strategies for RAG pipelines.
   (~6 min)

7. [Context Engineering](../docs/02-retrieval-augmented-generation/context-engineering.md)
   Prompt construction, context window management, retrieval integration,
   re-ranking, and techniques for maximizing answer quality from retrieved
   context.
   (~6 min)

8. [Agent Fundamentals](../docs/03-ai-agents/agent-fundamentals.md)
   Agent architectures, tool use patterns, the ReAct framework, planning
   strategies (task decomposition, reflection), and memory systems
   (short-term, long-term, episodic).
   (~9 min)

9. [Multi-Agent Architectures](../docs/03-ai-agents/multi-agent-architectures.md)
   Orchestration patterns (supervisor, peer-to-peer, hierarchical),
   communication protocols, task decomposition, error recovery, and
   frameworks for building multi-agent systems.
   (~9 min)

10. [Energy Efficiency](../docs/04-hardware-and-infrastructure/energy-efficiency.md)
    Power consumption profiles for training and inference, PUE metrics,
    carbon-aware scheduling, hardware efficiency trends, and optimization
    techniques for reducing the environmental footprint of LLM workloads.
    (~14 min)

11. [LLM Security and Safety](../docs/05-llm-security-and-safety/README.md)
    Overview of the security threat surface, prompt injection attack vectors,
    and alignment safety considerations.
    (~2 min)

**Total estimated reading time: ~95 minutes**

---

## Software Engineer (New to AI)

A foundations-first path that prioritizes practical understanding over deep
theory. This path starts with how models work at a high level, moves quickly
into the retrieval and agent patterns you are most likely to build with, and
defers hardware and infrastructure details to the end as optional reading.
It is designed for experienced software engineers who are new to AI/ML and
want to become productive with LLM-based application development.

**Why this order:** You need to understand what a model is before you can
use one effectively. Then the path follows the typical application
development flow: prepare data for retrieval, set up a vector store,
engineer the context, build agents, and understand security risks before
shipping.

1. [How LLMs Work](../docs/01-llm-foundations/how-llms-work.md)
   Start here. Understand tokenization, the Transformer architecture, and
   how text generation works end to end. Focus on building intuition rather
   than memorizing every equation.
   (~7 min)

2. [Training and Fine-Tuning](../docs/01-llm-foundations/training-and-fine-tuning.md)
   Learn how models are trained and adapted. This will help you understand
   what fine-tuning can and cannot do for your use case, and when to prefer
   prompting or RAG instead.
   (~8 min)

3. [Chunking Strategies](../docs/02-retrieval-augmented-generation/chunking-strategies.md)
   Practical guide to preparing documents for retrieval. This is the first
   step in building any RAG system, and chunking decisions directly affect
   retrieval quality.
   (~6 min)

4. [Vector Databases](../docs/02-retrieval-augmented-generation/vector-databases.md)
   How embeddings and similarity search work in practice, which vector
   databases to consider for different use cases, and how to evaluate
   indexing trade-offs.
   (~7 min)

5. [Context Engineering](../docs/02-retrieval-augmented-generation/context-engineering.md)
   How to assemble effective prompts with retrieved context, manage token
   budgets, handle context window limits, and improve answer quality through
   prompt structure.
   (~6 min)

6. [Agent Fundamentals](../docs/03-ai-agents/agent-fundamentals.md)
   How to build LLM-powered agents that use tools, plan multi-step tasks,
   and maintain conversational state. Covers the ReAct pattern and practical
   considerations for reliability.
   (~9 min)

7. [Multi-Agent Architectures](../docs/03-ai-agents/multi-agent-architectures.md)
   Patterns for composing multiple agents into larger systems, including
   supervisor and peer-to-peer topologies, and when multi-agent designs are
   worth the added complexity.
   (~9 min)

8. [LLM Security and Safety](../docs/05-llm-security-and-safety/README.md)
   Key security risks you need to be aware of before shipping LLM-powered
   features, including prompt injection, data leakage, and output safety.
   (~2 min)

9. [GPU Architecture for LLMs](../docs/04-hardware-and-infrastructure/gpu-architecture-for-llms.md)
   Optional but valuable: understand the hardware constraints that shape
   model behavior, latency, and cost. Helps you reason about why certain
   model sizes or quantization levels are preferred.
   (~12 min)

10. [Infrastructure and Scaling](../docs/04-hardware-and-infrastructure/infrastructure-and-scaling.md)
    Optional: serving infrastructure, scaling strategies, and deployment
    considerations for production LLM systems.
    (~15 min)

**Total estimated reading time: ~81 minutes
(core path without optional items: ~54 minutes)**

---

## Technical Leader / PM

A strategic overview path that focuses on capabilities, trade-offs, costs,
and decision-making. This path deliberately skips deep implementation details
(GPU memory hierarchies, chunking algorithms) and emphasizes the "what" and
"why" over the "how." It is designed for technical leaders, product managers,
and engineering managers who need to make informed decisions about LLM
adoption, architecture, and investment without writing the implementation
code themselves.

**Why this order:** Start with enough technical foundation to understand
the landscape, then move to the application patterns and cost structures
that drive strategic decisions.

1. [How LLMs Work](../docs/01-llm-foundations/how-llms-work.md)
   Build a mental model of what LLMs are and how they generate text. Focus
   on the conceptual flow (text in, tokens, attention, text out) rather
   than mathematical details. This vocabulary is essential for communicating
   with your engineering team.
   (~7 min)

2. [Training and Fine-Tuning](../docs/01-llm-foundations/training-and-fine-tuning.md)
   Understand the training pipeline and the decision framework for when
   fine-tuning is worth the investment versus prompting or RAG. Pay
   attention to the cost and data requirements of each approach.
   (~8 min)

3. [Context Engineering](../docs/02-retrieval-augmented-generation/context-engineering.md)
   Learn how the way you structure and select context shapes model output
   quality. This directly impacts product decisions about what information
   to retrieve, how much context to include, and where quality bottlenecks
   arise.
   (~6 min)

4. [Agent Fundamentals](../docs/03-ai-agents/agent-fundamentals.md)
   Understand what LLM-powered agents can do, where their reliability
   boundaries are, and how they fit into product architectures. Focus on
   capability assessment and risk factors rather than implementation
   details.
   (~9 min)

5. [Multi-Agent Architectures](../docs/03-ai-agents/multi-agent-architectures.md)
   Evaluate multi-agent patterns for complex workflows. Understand the
   trade-offs between single-agent and multi-agent designs, and when the
   added complexity and cost is justified by the business problem.
   (~9 min)

6. [Energy Efficiency](../docs/04-hardware-and-infrastructure/energy-efficiency.md)
   Understand the cost, power, and sustainability implications of LLM
   deployment at scale. This is increasingly relevant for budgeting,
   vendor selection, and corporate sustainability commitments.
   (~14 min)

7. [Infrastructure and Scaling](../docs/04-hardware-and-infrastructure/infrastructure-and-scaling.md)
   Skim for deployment models, cost drivers (GPU hours, memory, bandwidth),
   and scaling constraints. Focus on the sections about serving trade-offs
   and deployment patterns rather than parallelism implementation details.
   (~15 min)

8. [LLM Security and Safety](../docs/05-llm-security-and-safety/README.md)
   Be aware of the risk surface for LLM-powered products. Prompt injection,
   data privacy, and output safety are risks that need to be addressed in
   product planning and compliance reviews.
   (~2 min)

**Total estimated reading time: ~70 minutes**

---

## Researcher

A theory-heavy, papers-focused path that prioritizes architectural details,
scaling behavior, algorithmic innovations, and the systems research frontier.
This path assumes strong ML fundamentals (linear algebra, probability,
optimization) and focuses on the material most relevant to understanding the
current state of the art and identifying open research questions. Where the
documentation references specific papers, those are particularly worth
following up on.

**Why this order:** Start with model architecture and training theory, then
study the hardware and systems constraints that define the practical research
frontier, then examine how retrieval and agent patterns extend model
capabilities in ways that raise new research questions.

1. [How LLMs Work](../docs/01-llm-foundations/how-llms-work.md)
   Review Transformer internals, positional encodings (RoPE and extensions),
   attention variants (MHA, GQA, MQA), and the architectural evolution from
   the original Transformer to modern decoder-only designs.
   (~7 min)

2. [Training and Fine-Tuning](../docs/01-llm-foundations/training-and-fine-tuning.md)
   Study the Kaplan and Chinchilla scaling laws, the alignment pipeline
   (SFT, RLHF, DPO), and parameter-efficient approaches. Consider the open
   questions around scaling law extrapolation and post-training compute
   allocation.
   (~8 min)

3. [GPU Architecture for LLMs](../docs/04-hardware-and-infrastructure/gpu-architecture-for-llms.md)
   Understand the hardware constraints that drive algorithmic innovation:
   the memory hierarchy and bandwidth bottleneck, arithmetic intensity and
   the roofline model, Flash Attention as IO-aware algorithm design, and
   the theory of quantization error propagation.
   (~12 min)

4. [Infrastructure and Scaling](../docs/04-hardware-and-infrastructure/infrastructure-and-scaling.md)
   Parallelism strategies (tensor, pipeline, data, expert), speculative
   decoding as an algorithm design problem, PagedAttention and memory
   management, and the systems research frontier including disaggregated
   serving and prefill-decode separation.
   (~15 min)

5. [Energy Efficiency](../docs/04-hardware-and-infrastructure/energy-efficiency.md)
   Efficiency metrics and measurement methodologies, hardware trends
   (process node improvements, accelerator specialization), carbon-aware
   training scheduling, and the sustainability research agenda for
   large-scale AI systems.
   (~14 min)

6. [Vector Databases](../docs/02-retrieval-augmented-generation/vector-databases.md)
   ANN algorithms (HNSW graph traversal, IVF clustering, product
   quantization), distance metrics and their geometric properties, and the
   theoretical trade-offs between recall, latency, and memory in
   high-dimensional search.
   (~7 min)

7. [Chunking Strategies](../docs/02-retrieval-augmented-generation/chunking-strategies.md)
   Semantic chunking methods, the relationship between chunk boundaries and
   retrieval quality, and the information-theoretic considerations
   underlying document segmentation for retrieval.
   (~6 min)

8. [Context Engineering](../docs/02-retrieval-augmented-generation/context-engineering.md)
   Context window utilization research, the "lost in the middle" phenomenon,
   retrieval-generation interaction dynamics, and how context placement
   affects model attention patterns and output quality.
   (~6 min)

9. [Agent Fundamentals](../docs/03-ai-agents/agent-fundamentals.md)
   Formal agent frameworks, planning algorithms (task decomposition, tree
   search, reflection), tool-use mechanisms and their theoretical grounding,
   and memory architectures for persistent agent state.
   (~9 min)

10. [Multi-Agent Architectures](../docs/03-ai-agents/multi-agent-architectures.md)
    Multi-agent communication protocols, coordination strategies
    (centralized vs. decentralized), emergent behavior in agent collectives,
    and connections to multi-agent systems research in classical AI.
    (~9 min)

11. [LLM Security and Safety](../docs/05-llm-security-and-safety/README.md)
    Adversarial robustness of language models, alignment failure modes, the
    theoretical limits of prompt injection defense, and open safety research
    questions.
    (~2 min)

**Total estimated reading time: ~95 minutes**
