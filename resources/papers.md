# Key Papers

> A curated reading list of the most important papers for understanding LLMs, RAG, and AI agents. Each entry includes a one-line takeaway to help you prioritize.

## Table of Contents
- [LLM Foundations](#llm-foundations)
- [Training & Alignment](#training--alignment)
- [Retrieval-Augmented Generation](#retrieval-augmented-generation)
- [AI Agents](#ai-agents)
- [Scaling & Emergent Abilities](#scaling--emergent-abilities)

---

## LLM Foundations

| Paper | Authors | Year | Key Takeaway |
|---|---|---|---|
| [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | Vaswani et al. | 2017 | Introduced the Transformer architecture — the foundation of all modern LLMs |
| [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805) | Devlin et al. | 2018 | Demonstrated that bidirectional pre-training produces powerful language representations for NLU |
| [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) | Brown et al. (GPT-3) | 2020 | Showed that scaling to 175B parameters enables few-shot learning without fine-tuning |
| [A Survey of Large Language Models](https://arxiv.org/abs/2303.18223) | Zhao et al. | 2023 | Comprehensive survey covering pre-training, adaptation, utilization, and evaluation of LLMs |
| [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971) | Touvron et al. | 2023 | Proved that smaller models trained on more data can match larger models — catalyzed open-source LLMs |

## Training & Alignment

| Paper | Authors | Year | Key Takeaway |
|---|---|---|---|
| [Training Language Models to Follow Instructions with Human Feedback](https://arxiv.org/abs/2203.02155) | Ouyang et al. (InstructGPT) | 2022 | Introduced RLHF for aligning LLMs with human intent — the recipe behind ChatGPT |
| [Direct Preference Optimization: Your Language Model Is Secretly a Reward Model](https://arxiv.org/abs/2305.18290) | Rafailov et al. | 2023 | Simplified RLHF by eliminating the reward model — train directly on preference pairs |
| [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073) | Bai et al. | 2022 | Model self-critiques against principles (RLAIF) — scalable alignment without human labelers |
| [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) | Hu et al. | 2021 | Parameter-efficient fine-tuning that reduces memory by 10–100x — made fine-tuning accessible |
| [LIMA: Less Is More for Alignment](https://arxiv.org/abs/2305.11206) | Zhou et al. | 2023 | Just 1,000 curated examples produce competitive SFT results — data quality > quantity |

## Retrieval-Augmented Generation

| Paper | Authors | Year | Key Takeaway |
|---|---|---|---|
| [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) | Lewis et al. | 2020 | The original RAG paper — combining retrieval with generation for knowledge-grounded responses |
| [Evaluating Chunking Strategies for Retrieval](https://research.trychroma.com/evaluating-chunking) | Chroma Research | 2024 | 200-token chunks with no overlap outperform popular defaults across all strategies |
| [Context Rot](https://research.trychroma.com/context-rot) | Chroma Research | 2024 | LLM performance degrades with context length; how context is presented matters more than what's included |
| [Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172) | Liu et al. | 2023 | Models attend more to the beginning and end of context — information in the middle is often missed |
| [HNSW: Efficient and Robust Approximate Nearest Neighbor](https://arxiv.org/abs/1603.09320) | Malkov & Yashunin | 2018 | The ANN algorithm used by most vector databases — hierarchical graph for fast similarity search |

## AI Agents

| Paper | Authors | Year | Key Takeaway |
|---|---|---|---|
| [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) | Yao et al. | 2022 | Interleaving reasoning traces with actions — the foundational pattern for LLM agents |
| [Toolformer: Language Models Can Teach Themselves to Use Tools](https://arxiv.org/abs/2302.04761) | Schick et al. | 2023 | LLMs can learn when and how to use tools (search, calculator, etc.) through self-supervised training |
| [Generative Agents: Interactive Simulacra of Human Behavior](https://arxiv.org/abs/2304.03442) | Park et al. | 2023 | Agents with memory, reflection, and planning simulate believable human behavior in a sandbox |
| [AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation](https://arxiv.org/abs/2308.08155) | Wu et al. | 2023 | Framework for multi-agent conversations with customizable agents and human-in-the-loop |
| [A Survey on Large Language Model Based Autonomous Agents](https://arxiv.org/abs/2308.11432) | Wang et al. | 2023 | Comprehensive survey of agent architectures: perception, reasoning, action, and memory |
| [MetaGPT: Meta Programming for Multi-Agent Collaborative Framework](https://arxiv.org/abs/2308.00352) | Hong et al. | 2023 | Assigns human-like roles (PM, architect, engineer) to agents for collaborative software development |

## Scaling & Emergent Abilities

| Paper | Authors | Year | Key Takeaway |
|---|---|---|---|
| [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) | Kaplan et al. | 2020 | LLM performance follows predictable power laws across parameters, data, and compute |
| [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556) | Hoffmann et al. (Chinchilla) | 2022 | Optimal training balances model size and data — most models were over-parameterized and under-trained |
| [Emergent Abilities of Large Language Models](https://arxiv.org/abs/2206.07682) | Wei et al. | 2022 | Certain capabilities appear suddenly at scale — fundamentally changes what larger models can do |
| [Are Emergent Abilities of Large Language Models a Mirage?](https://arxiv.org/abs/2304.15004) | Schaeffer et al. | 2023 | Some "emergent" abilities may be measurement artifacts — important counterpoint to the emergence narrative |
| [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903) | Wei et al. | 2022 | Adding "let's think step by step" dramatically improves reasoning — but only in large models |
