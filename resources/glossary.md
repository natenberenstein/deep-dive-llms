# Glossary

This glossary provides concise definitions for key terms encountered throughout
the deep-dive-llms documentation. Terms are organized alphabetically for quick
reference. Where applicable, each entry links to the most relevant document in
this repository for further reading.

---

## A

**Attention** -- The core mechanism in Transformer models that allows each
token in a sequence to weigh and aggregate information from all other tokens.
Attention computes a weighted sum of value vectors, where the weights are
derived from the compatibility (dot product) between query and key vectors.
The standard formulation is Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V.
See [How LLMs Work](../docs/01-llm-foundations/how-llms-work.md).

**Autoregressive** -- A generation strategy in which a model produces output
one token at a time, conditioning each new token on all previously generated
tokens. Most modern LLMs, including the GPT and LLaMA families, are
autoregressive language models. This sequential dependency is what makes
inference memory-bandwidth-bound rather than compute-bound.
See [How LLMs Work](../docs/01-llm-foundations/how-llms-work.md).

---

## B

**BPE (Byte-Pair Encoding)** -- A subword tokenization algorithm that
iteratively merges the most frequent pair of adjacent characters or character
sequences in a corpus. BPE balances vocabulary size against the ability to
represent rare or out-of-vocabulary words. It is used by models such as GPT-2,
GPT-4, and LLaMA. Variants like byte-level BPE operate directly on raw bytes,
eliminating the need for Unicode pre-processing.
See [How LLMs Work](../docs/01-llm-foundations/how-llms-work.md).

---

## C

**Chain-of-Thought (CoT)** -- A prompting technique that encourages a language
model to produce intermediate reasoning steps before arriving at a final
answer. Chain-of-thought prompting has been shown to improve performance on
arithmetic, commonsense, and symbolic reasoning tasks, especially in larger
models. The technique can be elicited either through explicit instructions
("think step by step") or by providing worked examples.

**Chinchilla Scaling** -- A set of scaling law findings from DeepMind
(Hoffmann et al., 2022) showing that many large language models were
significantly over-parameterized relative to their training data. The
Chinchilla-optimal regime suggests that compute budgets should be split
roughly equally between increasing model parameters and increasing training
tokens. This finding reshaped how the field allocates training compute.
See [Training and Fine-Tuning](../docs/01-llm-foundations/training-and-fine-tuning.md).

**Constitutional AI (CAI)** -- An alignment approach developed by Anthropic
in which a model is guided by a set of written principles (a "constitution")
during training. The model critiques and revises its own outputs according
to those principles, reducing reliance on large-scale human feedback for
identifying harmful content. CAI combines supervised learning on self-revised
outputs with reinforcement learning from AI feedback (RLAIF).

**Context Window** -- The maximum number of tokens a model can process in a
single forward pass, encompassing both the input prompt and the generated
output. Context window sizes have grown from 2,048 tokens in GPT-3 to
128,000 or more in recent models such as GPT-4 Turbo and Claude 3. The
effective use of the context window is a key concern in RAG systems and
prompt engineering.
See [Context Engineering](../docs/02-retrieval-augmented-generation/context-engineering.md).

**Cross-Encoder** -- A neural architecture that takes a pair of texts (such
as a query and a document) as a single concatenated input and produces a
relevance score. Cross-encoders are more accurate than bi-encoders for
ranking but are computationally expensive because they cannot pre-compute
document representations independently. They are typically used in the
re-ranking stage of a retrieval pipeline rather than for initial candidate
retrieval.
See [Vector Databases](../docs/02-retrieval-augmented-generation/vector-databases.md).

---

## D

**DPO (Direct Preference Optimization)** -- A fine-tuning method that aligns
language models to human preferences without training a separate reward model.
DPO reformulates the RLHF objective as a classification loss over preference
pairs, treating the language model itself as an implicit reward model. This
makes alignment training simpler, more stable, and less computationally
expensive than standard RLHF with PPO.
See [Training and Fine-Tuning](../docs/01-llm-foundations/training-and-fine-tuning.md).

---

## E

**Embedding** -- A dense, fixed-dimensional vector representation of a token,
sentence, or document in continuous space. Embeddings capture semantic
similarity such that related concepts are positioned near each other in the
vector space. In the context of RAG, embedding models convert both queries
and documents into vectors for similarity-based retrieval. Common embedding
dimensions range from 384 to 3,072 depending on the model.
See [Vector Databases](../docs/02-retrieval-augmented-generation/vector-databases.md).

**Emergent Abilities** -- Capabilities that appear in language models only
above certain scale thresholds and are not present or predictable in smaller
models. Frequently cited examples include multi-step arithmetic, code
generation, and chain-of-thought reasoning. The existence and definition of
emergent abilities remains an active area of debate, with some researchers
arguing the phenomenon is an artifact of discontinuous evaluation metrics
rather than a genuine phase transition.

---

## F

**Few-Shot** -- A prompting paradigm in which the model is given a small
number of input-output examples (typically 2-10) within the prompt to
demonstrate the desired task format and expected behavior. Few-shot prompting
leverages in-context learning and often substantially outperforms zero-shot
prompting, particularly on tasks that require specific output formatting or
domain conventions.
See [How LLMs Work](../docs/01-llm-foundations/how-llms-work.md).

**Fine-Tuning** -- The process of continuing the training of a pre-trained
model on a smaller, task-specific or domain-specific dataset. Fine-tuning
updates the model weights to improve performance on targeted tasks while
retaining general capabilities learned during pre-training. Full fine-tuning
updates all parameters, whereas parameter-efficient methods such as LoRA
update only a small fraction.
See [Training and Fine-Tuning](../docs/01-llm-foundations/training-and-fine-tuning.md).

**Flash Attention** -- An IO-aware, exact attention algorithm developed by
Dao et al. that reduces memory usage from quadratic to linear in sequence
length by tiling the computation and avoiding materialization of the full
attention matrix in GPU HBM. Flash Attention exploits the GPU memory hierarchy
(SRAM vs. HBM) to achieve significant wall-clock speedups (2-4x) without
any approximation or change to the model's output.
See [GPU Architecture for LLMs](../docs/04-hardware-and-infrastructure/gpu-architecture-for-llms.md).

---

## G

**GGUF (GPT-Generated Unified Format)** -- A binary file format for storing
quantized language model weights, designed for efficient inference on consumer
hardware. GGUF is the successor to the GGML format and is widely used with
llama.cpp and other local inference frameworks. It supports multiple
quantization levels (e.g., Q4_K_M, Q5_K_S) and stores all model metadata,
tokenizer data, and weights in a single self-contained file.

**Grounding** -- The practice of anchoring a language model's responses in
verifiable external sources such as retrieved documents, databases, or API
outputs. Grounding reduces hallucination and improves factual reliability by
giving the model specific evidence to reference rather than relying solely on
parametric knowledge. RAG is one of the primary techniques for grounding.
See [Context Engineering](../docs/02-retrieval-augmented-generation/context-engineering.md).

**Guardrails** -- Input and output filters, validation rules, or secondary
model calls that constrain a language model's behavior to stay within
acceptable boundaries. Guardrails can prevent generation of harmful content,
enforce output format requirements, block prompt injection attempts, or
ensure that responses remain within a defined scope. They may be implemented
as rule-based systems, classifier models, or a combination of both.

---

## H

**Hallucination** -- The generation of content that is fluent and
plausible-sounding but factually incorrect, unsupported by the source
material, or entirely fabricated. Hallucination is one of the most significant
reliability challenges in deploying LLMs. Mitigation strategies include
grounding with retrieved evidence, constrained decoding, calibration
techniques, and evaluation frameworks that detect unfaithful content.

**HyDE (Hypothetical Document Embeddings)** -- A retrieval technique in which
the language model first generates a hypothetical answer to a query, and then
that hypothetical answer is embedded and used to search the vector store.
HyDE can improve retrieval quality when queries are short, abstract, or
phrased differently from the documents, because the generated hypothetical
document is often closer in embedding space to relevant real documents than
the original query would be.

---

## I

**In-Context Learning (ICL)** -- The ability of a language model to learn and
perform new tasks solely from examples or instructions provided within the
input prompt, without any gradient updates or fine-tuning. In-context learning
is the mechanism underlying few-shot and zero-shot prompting. The theoretical
basis for ICL remains an active research area, with hypotheses ranging from
implicit Bayesian inference to implicit gradient descent.

**Inference** -- The process of running a trained model to generate predictions
or outputs for new inputs. In the context of LLMs, inference involves iterative
autoregressive token generation, and its cost is dominated by memory bandwidth
rather than raw compute (making it "memory-bound"). Optimizing inference
throughput and latency is a primary concern for serving LLMs at scale.
See [GPU Architecture for LLMs](../docs/04-hardware-and-infrastructure/gpu-architecture-for-llms.md).

**Instruction Tuning** -- A form of fine-tuning in which a model is trained on
datasets of (instruction, response) pairs to improve its ability to follow
natural language instructions across a wide range of tasks. Instruction tuning
is a key step that transforms a base language model (which merely predicts
next tokens) into a useful assistant that can follow directions. Notable
instruction-tuned datasets include FLAN, Alpaca, and OpenAssistant.
See [Training and Fine-Tuning](../docs/01-llm-foundations/training-and-fine-tuning.md).

---

## K

**KV Cache (Key-Value Cache)** -- A memory optimization for autoregressive
inference that stores the key and value tensors computed during previous token
generation steps so they do not need to be recomputed at each subsequent step.
The KV cache grows linearly with sequence length, number of layers, and batch
size. Managing its memory footprint is a central challenge in serving
long-context models, and techniques like PagedAttention and grouped-query
attention have been developed specifically to address this bottleneck.
See [GPU Architecture for LLMs](../docs/04-hardware-and-infrastructure/gpu-architecture-for-llms.md).

---

## L

**LoRA (Low-Rank Adaptation)** -- A parameter-efficient fine-tuning method
that freezes the pre-trained model weights and injects trainable low-rank
decomposition matrices (A and B, where the update is W + BA) into targeted
layers of the Transformer. LoRA drastically reduces the number of trainable
parameters (often by 10,000x) and GPU memory required for fine-tuning while
maintaining competitive performance. Multiple LoRA adapters can be swapped
at serving time without reloading the base model.
See [Training and Fine-Tuning](../docs/01-llm-foundations/training-and-fine-tuning.md).

---

## M

**MMLU (Massive Multitask Language Understanding)** -- A benchmark consisting
of multiple-choice questions across 57 academic subjects, ranging from
elementary mathematics to professional law and medicine. MMLU is widely used
to evaluate the breadth of knowledge and reasoning capabilities of language
models. While it remains a popular benchmark, researchers have noted
limitations including answer contamination in training data and the narrow
scope of multiple-choice evaluation.

**Multi-Head Attention (MHA)** -- A variant of the attention mechanism that
runs multiple attention operations ("heads") in parallel, each with its own
learned query, key, and value projection matrices. The outputs of all heads
are concatenated and linearly projected, allowing the model to jointly attend
to information from different representation subspaces at different positions.
Standard Transformer models use 8 to 128 attention heads depending on model
size.
See [How LLMs Work](../docs/01-llm-foundations/how-llms-work.md).

---

## P

**PagedAttention** -- A memory management technique for KV caches, introduced
by the vLLM project, that borrows ideas from virtual memory paging in
operating systems. PagedAttention stores KV cache blocks in non-contiguous
GPU memory pages rather than requiring a single contiguous allocation per
sequence. This reduces memory waste from internal fragmentation by up to 55%
and enables significantly higher batch sizes during serving.
See [Infrastructure and Scaling](../docs/04-hardware-and-infrastructure/infrastructure-and-scaling.md).

**Parameter-Efficient Fine-Tuning (PEFT)** -- A family of methods that
fine-tune only a small subset of a model's parameters (or add a small number
of new parameters) rather than updating the entire model. PEFT techniques
include LoRA, QLoRA, adapters, prefix tuning, and prompt tuning. They reduce
compute and memory costs dramatically and make fine-tuning of large models
feasible on consumer-grade hardware. The trade-off is typically a small
reduction in maximum achievable quality compared to full fine-tuning.
See [Training and Fine-Tuning](../docs/01-llm-foundations/training-and-fine-tuning.md).

**Perplexity** -- A standard evaluation metric for language models, defined
as the exponentiated average negative log-likelihood of a test set. Lower
perplexity indicates that the model assigns higher probability to the observed
text. Perplexity measures how "surprised" the model is by the data, but it
does not directly measure the usefulness or safety of generated outputs.

**Prompt Injection** -- An adversarial attack in which malicious instructions
are embedded within user input (or within retrieved context in a RAG system)
to override the model's system prompt or safety guidelines. Prompt injection
is a significant security concern for deployed LLM applications and remains
an open problem without a complete solution. Defense strategies include input
sanitization, instruction hierarchy, and secondary classifier models.

---

## Q

**QLoRA (Quantized Low-Rank Adaptation)** -- An extension of LoRA that
quantizes the frozen base model weights to 4-bit precision (using the NF4
data type) while keeping the LoRA adapter weights in higher precision
(typically bfloat16). QLoRA introduces double quantization and paged
optimizers to further reduce memory overhead. It enables fine-tuning of
large models (e.g., 65B parameters) on a single 48GB consumer GPU with
minimal quality degradation compared to full 16-bit fine-tuning.
See [Training and Fine-Tuning](../docs/01-llm-foundations/training-and-fine-tuning.md).

**Quantization** -- The process of reducing the numerical precision of model
weights and/or activations (e.g., from 16-bit floating point to 8-bit or
4-bit integers) to decrease memory footprint and improve inference throughput.
Common approaches include post-training quantization (PTQ), which quantizes
a trained model without additional training, and quantization-aware training
(QAT), which simulates quantization effects during training. Quantization
involves trade-offs between model quality and resource efficiency.
See [GPU Architecture for LLMs](../docs/04-hardware-and-infrastructure/gpu-architecture-for-llms.md).

---

## R

**RAG (Retrieval-Augmented Generation)** -- An architecture that enhances
language model generation by first retrieving relevant documents from an
external knowledge base and then including them in the model's context. RAG
reduces hallucination, enables access to up-to-date or proprietary
information, and supports domain-specific applications without requiring
fine-tuning. A typical RAG pipeline consists of document ingestion, chunking,
embedding, indexing, retrieval, optional re-ranking, and generation.
See [Chunking Strategies](../docs/02-retrieval-augmented-generation/chunking-strategies.md)
and [Context Engineering](../docs/02-retrieval-augmented-generation/context-engineering.md).

**RAGAS (Retrieval-Augmented Generation Assessment)** -- A framework for
evaluating RAG pipelines using automated metrics such as faithfulness (is the
answer supported by the retrieved context?), answer relevance (does the answer
address the question?), and context precision (are the retrieved documents
relevant?). RAGAS provides evaluation without requiring hand-labeled
ground-truth answers for every query, making it practical for iterating on
production RAG systems.

**ReAct (Reasoning + Acting)** -- A prompting and agent framework that
interleaves chain-of-thought reasoning with tool-use actions. The model
alternates between generating reasoning traces ("Thought"), executing actions
("Action"), and processing results ("Observation"), using observations from
each action to inform subsequent reasoning steps. ReAct has become a
foundational pattern for building LLM-powered agents.
See [Agent Fundamentals](../docs/03-ai-agents/agent-fundamentals.md).

**Re-ranking** -- A second-stage retrieval step that takes an initial set of
candidate documents (typically from a fast, approximate retrieval method such
as ANN search) and re-scores them using a more accurate but computationally
expensive model, such as a cross-encoder. Re-ranking improves the precision
of the final result set and is especially valuable when the initial retrieval
returns a noisy set of candidates.
See [Vector Databases](../docs/02-retrieval-augmented-generation/vector-databases.md).

**RLHF (Reinforcement Learning from Human Feedback)** -- A training
methodology that aligns language model behavior with human preferences. The
process involves three stages: (1) supervised fine-tuning on demonstration
data, (2) training a reward model on human comparison data (which response
is better?), and (3) optimizing the language model against that reward model
using reinforcement learning, typically PPO. RLHF is a core component of the
alignment pipeline for models like ChatGPT and Claude.
See [Training and Fine-Tuning](../docs/01-llm-foundations/training-and-fine-tuning.md).

**RoPE (Rotary Position Embedding)** -- A positional encoding scheme that
encodes absolute position using rotation matrices applied to query and key
vectors, while naturally incorporating relative position information in the
attention dot product. RoPE enables better length generalization than learned
absolute positional embeddings and is used in LLaMA, Mistral, Qwen, and many
other modern architectures. Extensions like YaRN and NTK-aware scaling allow
RoPE models to extrapolate beyond their training context length.
See [How LLMs Work](../docs/01-llm-foundations/how-llms-work.md).

---

## S

**Scaling Laws** -- Empirical relationships, first systematically
characterized by Kaplan et al. (2020) and refined by Hoffmann et al. (2022),
that describe how language model performance (measured by cross-entropy loss)
improves as a power law with increases in model parameters, dataset size, and
compute budget. Scaling laws guide decisions about how to allocate resources
across model size and training duration, and they underpin the strategic
planning of frontier model development.
See [Training and Fine-Tuning](../docs/01-llm-foundations/training-and-fine-tuning.md).

**Self-Attention** -- A specific form of attention in which the queries, keys,
and values all come from the same input sequence. Self-attention allows each
position in a sequence to attend to every other position, enabling the model
to capture long-range dependencies regardless of distance. It is the primary
building block of the Transformer architecture, replacing the recurrent
connections used in earlier sequence models like LSTMs and GRUs.
See [How LLMs Work](../docs/01-llm-foundations/how-llms-work.md).

**SFT (Supervised Fine-Tuning)** -- The stage of post-training in which a
pre-trained language model is fine-tuned on curated (prompt, response) pairs
using standard supervised learning with a cross-entropy loss. SFT typically
precedes RLHF in the alignment pipeline and teaches the model the basic
format, style, and behavioral expectations of helpful responses. The quality
of SFT data is considered more important than its quantity.
See [Training and Fine-Tuning](../docs/01-llm-foundations/training-and-fine-tuning.md).

**Speculative Decoding** -- An inference acceleration technique in which a
smaller, faster "draft" model proposes multiple candidate tokens in a single
step, and the larger target model then verifies all of them in parallel with
a single forward pass. Correct tokens are accepted while incorrect ones are
rejected and regenerated. Speculative decoding can reduce latency by 2-3x
without changing the output distribution of the target model.
See [Infrastructure and Scaling](../docs/04-hardware-and-infrastructure/infrastructure-and-scaling.md).

---

## T

**Temperature** -- A hyperparameter that controls the randomness of token
sampling during generation. Temperature scales the logits (raw model outputs)
before the softmax function: values below 1.0 sharpen the probability
distribution, making output more deterministic and focused; values above 1.0
flatten the distribution, making output more diverse and creative. A
temperature of 0 produces greedy decoding, always selecting the
highest-probability token.

**Token** -- The fundamental unit of text that a language model processes. A
token may be a word, a subword fragment, a single character, or a byte
sequence, depending on the tokenizer used. Common tokenizers produce tokens
that average roughly 3-4 characters in English text. The number of tokens in
an input determines memory usage, compute cost, and whether the text fits
within the model's context window.
See [How LLMs Work](../docs/01-llm-foundations/how-llms-work.md).

**Tokenizer** -- The component that converts raw text into a sequence of
token IDs (integers) from a fixed vocabulary and can reverse the mapping
(decoding). Tokenizers are trained on large text corpora separately from the
model itself and significantly influence model behavior, computational
efficiency, and multilingual capability. Common tokenization algorithms
include BPE, WordPiece, Unigram, and SentencePiece.
See [How LLMs Work](../docs/01-llm-foundations/how-llms-work.md).

**Top-K / Top-P (Nucleus Sampling)** -- Decoding strategies that restrict
the set of candidate tokens considered at each generation step to reduce the
probability of sampling low-quality tokens. Top-K sampling considers only the
K most probable tokens. Top-P (nucleus) sampling considers the smallest set
of tokens whose cumulative probability exceeds the threshold P. Both methods
can be combined with temperature scaling and are commonly used together in
practice.

**Transformer** -- The neural network architecture introduced by Vaswani et
al. (2017) in the paper "Attention Is All You Need." The Transformer replaces
recurrence and convolution with multi-head self-attention and position-wise
feed-forward layers. It enables efficient parallelization during training and
forms the backbone of virtually all modern large language models. Variants
include encoder-only (BERT), decoder-only (GPT), and encoder-decoder (T5)
configurations.
See [How LLMs Work](../docs/01-llm-foundations/how-llms-work.md).

---

## V

**Vector Database** -- A specialized database system designed to store, index,
and efficiently query high-dimensional embedding vectors. Vector databases use
approximate nearest neighbor (ANN) algorithms such as HNSW, IVF, and product
quantization to enable fast similarity search at scale. They are a critical
infrastructure component of RAG pipelines and other embedding-based retrieval
systems. Examples include Pinecone, Weaviate, Milvus, Qdrant, and pgvector.
See [Vector Databases](../docs/02-retrieval-augmented-generation/vector-databases.md).

**Vision-Language Model (VLM)** -- A multimodal model that can process both
visual (image or video) and textual inputs within a unified architecture,
enabling tasks such as image captioning, visual question answering, document
understanding, and image-guided reasoning. VLMs typically use a vision encoder
(e.g., ViT) connected to a language model through a projection layer or
cross-attention mechanism. Examples include GPT-4V, Claude 3 (with vision),
LLaVA, and Gemini.

---

## Z

**Zero-Shot** -- A prompting paradigm in which the model is asked to perform
a task using only a natural language instruction, without any demonstration
examples provided in the prompt. Zero-shot performance is a measure of a
model's ability to generalize from its pre-training and instruction-tuning to
new task formats that it may not have explicitly seen during training.
See [How LLMs Work](../docs/01-llm-foundations/how-llms-work.md).
