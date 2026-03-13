# AI Agents

> Moving beyond single-turn Q&A — how LLMs become autonomous agents that reason, plan, use tools, and collaborate.

## What This Section Covers

AI agents represent the next evolution in LLM applications. Instead of a single prompt-response cycle, agents operate in loops — observing, reasoning, acting, and learning from the results. This section covers the fundamental patterns for building individual agents and the architectural patterns for coordinating multiple agents.

## Agent Architecture Patterns

```mermaid
graph TD
    subgraph Single Agent
        A[Perception<br/>User input, tool results, memory] --> B[Reasoning<br/>Plan, reflect, decide]
        B --> C[Action<br/>Generate text, call tools, update memory]
        C --> A
    end

    subgraph Multi-Agent
        D[Supervisor Agent] --> E[Research Agent]
        D --> F[Writer Agent]
        D --> G[Reviewer Agent]
        E --> D
        F --> D
        G --> D
    end

    style B fill:#e06c75,stroke:#a84e55,color:#fff
    style D fill:#e06c75,stroke:#a84e55,color:#fff
```

## Pages in This Section

| Page | What You'll Learn |
|---|---|
| [Agent Fundamentals](agent-fundamentals.md) | What makes an LLM an "agent," ReAct pattern, tool use, memory, and planning |
| [Tool Design Patterns](tool-design-patterns.md) | API wrapping, JSON Schema best practices, error handling, composition, and safety |
| [Memory and State Management](memory-and-state-management.md) | Conversation memory strategies, long-term memory stores, episodic and procedural memory, state machines, and checkpointing |
| [Multi-Agent Architectures](multi-agent-architectures.md) | Single vs. multi-agent tradeoffs, common patterns, LangChain, and LangGraph |
| [Agent Frameworks](agent-frameworks.md) | Claude Agent SDK, OpenAI Assistants, LangGraph, CrewAI, AutoGen compared |
| [Agent Evaluation and Safety](agent-evaluation-and-safety.md) | Evaluation dimensions, guardrails, sandboxing, failure modes, and monitoring |

## Suggested Reading Order

1. Start with **Agent Fundamentals** to understand the core concepts: reasoning loops, tool use, and memory
2. Then read **Tool Design Patterns** to learn how to build reliable tools for agents
3. Then read **Memory and State Management** to understand how agents retain context and manage execution state
4. Then read **Multi-Agent Architectures** to learn how to compose agents into systems
5. Then read **Agent Frameworks** to compare the available frameworks and choose one
6. Finally, **Agent Evaluation and Safety** for testing, guardrails, and production monitoring
