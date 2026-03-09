# Dataset Preparation

> **TL;DR:** Data quality is the single biggest determinant of fine-tuning success. 500 carefully curated examples often outperform 50,000 noisy ones. Your training data should be in instruction-response or chat format, deduplicated, cleaned of contradictions, and representative of the distribution you want the model to learn. Collection strategies range from manual curation (highest quality, lowest scale) to synthetic generation (lowest quality, highest scale). Invest in annotation tools and quality review processes before scaling data collection.

## Table of Contents
- [Why This Matters](#why-this-matters)
- [Data Quality Over Quantity](#data-quality-over-quantity)
- [Data Formats](#data-formats)
- [Collection Strategies](#collection-strategies)
- [Cleaning and Preprocessing](#cleaning-and-preprocessing)
- [Common Pitfalls](#common-pitfalls)
- [Tools and Infrastructure](#tools-and-infrastructure)
- [Key Takeaways](#key-takeaways)
- [References](#references)

## Why This Matters

Fine-tuning is only as good as the data you train on. A perfect LoRA configuration with a perfect base model will produce a terrible fine-tuned model if the training data is noisy, inconsistent, or unrepresentative. Conversely, clean and well-structured data can compensate for suboptimal hyperparameters.

Most fine-tuning failures trace back to data problems:
- Contradictory examples that confuse the model
- Distribution mismatch between training data and real-world queries
- Insufficient diversity leading to brittle, overfit behavior
- Quality issues (typos, incorrect labels, truncated responses) that propagate into model outputs

Building a high-quality dataset is the most time-consuming and most valuable part of the fine-tuning process.

## Data Quality Over Quantity

### The LIMA Insight

The LIMA paper (Zhou et al., 2023) demonstrated that a model fine-tuned on just 1,000 carefully curated examples performed comparably to models trained on 52,000 examples (Alpaca) or 70,000 examples (ShareGPT). The key difference was quality: every LIMA example was manually reviewed and refined.

### What "Quality" Means

**Correctness** -- Every response must be factually accurate and logically sound. A single incorrect example can teach the model to repeat that error.

**Consistency** -- All examples should follow the same formatting conventions, tone, and style. Mixed styles confuse the model about what output format you want.

**Completeness** -- Responses should be thorough enough to be useful. Truncated or partial responses teach the model to stop early.

**Representativeness** -- The training distribution should match the real-world query distribution. If 80% of production queries are about topic A, your training data should reflect that.

**Difficulty gradient** -- Include examples ranging from simple to complex. All-easy or all-hard datasets produce models that fail on the other end.

### How Much Data Do You Need?

| Task Complexity | Minimum Examples | Recommended | Notes |
|---|---|---|---|
| Style/tone change | 100-200 | 500 | Base model already capable, just adjusting format |
| Output format enforcement | 200-500 | 1,000 | Consistent structured output (JSON, tables) |
| Domain-specific behavior | 500-1,000 | 2,000-5,000 | Teaching new domain conventions |
| Complex reasoning tasks | 1,000-5,000 | 5,000-10,000 | Significant behavior change from base |
| New language/domain | 5,000-10,000 | 10,000-50,000 | Teaching fundamentally new capabilities |

These are guidelines, not rules. The minimum depends on how far the desired behavior is from the base model's default.

## Data Formats

### Instruction-Response Format

The simplest format: a single instruction paired with the expected response.

```json
{
  "instruction": "Summarize the following customer complaint in 2-3 sentences, identifying the core issue and sentiment.",
  "input": "I ordered a laptop on March 1st and was told it would arrive by March 5th. It's now March 12th and I still haven't received it. Every time I call customer service, I get transferred to a different person who tells me something different. I just want my laptop or my money back.",
  "output": "The customer ordered a laptop on March 1st with an expected delivery by March 5th, but has not received it as of March 12th. Their core issue is the delivery delay compounded by inconsistent information from customer service. The sentiment is frustrated but reasonable, seeking either delivery or a refund."
}
```

**Best for:** Single-turn tasks like classification, summarization, extraction, and reformatting.

### Chat/Conversation Format

Multi-turn conversations with role-based messages. Essential for conversational fine-tuning.

```json
{
  "messages": [
    {"role": "system", "content": "You are a technical support agent for CloudStack, a cloud hosting platform. Be concise and helpful. Always suggest checking the documentation first."},
    {"role": "user", "content": "My deployment keeps failing with a timeout error."},
    {"role": "assistant", "content": "Timeout errors during deployment usually indicate one of three issues:\n\n1. Build step exceeding the 15-minute limit\n2. Health check failing after deployment\n3. Network connectivity issues during dependency installation\n\nCould you share the deployment logs? You can find them in your dashboard under Deployments > [Failed Deployment] > Logs."},
    {"role": "user", "content": "The logs show the health check is timing out."},
    {"role": "assistant", "content": "That confirms it is a health check issue. Check these items:\n\n1. Verify your app listens on the port specified in your configuration (default: 8080)\n2. Ensure your health endpoint (usually /health or /) returns a 200 status code within 30 seconds\n3. Check that your app starts up within the 60-second startup grace period\n\nThe most common cause is the app binding to localhost instead of 0.0.0.0. See our documentation: docs.cloudstack.com/health-checks"}
  ]
}
```

**Best for:** Chatbots, assistants, and any multi-turn application where conversation history matters.

### Completion Format

Raw text continuation -- the model learns to complete partial text. Less structured but useful for specific applications.

```json
{
  "text": "### Legal Brief Summary\n\nCase: Smith v. Johnson (2024)\nCourt: Federal District Court, Northern District\n\nFacts: The plaintiff alleges that the defendant breached a non-compete agreement by accepting employment at a competing firm within the restricted geographic area and time period.\n\nHolding: The court found in favor of the plaintiff, ruling that the non-compete clause was reasonable in scope and duration.\n\nSignificance: This case reinforces the enforceability of non-compete agreements when they are narrowly tailored to protect legitimate business interests."
}
```

**Best for:** Domain-specific text generation where you want the model to internalize a writing style, terminology, or structure.

### Template Considerations

Different base models expect different chat templates:

| Model Family | Template Style | Special Tokens |
|---|---|---|
| LLaMA 2/3 | `[INST]...[/INST]` | `<s>`, `</s>` |
| Mistral | `[INST]...[/INST]` | `<s>`, `</s>` |
| ChatML (many models) | `<\|im_start\|>role...` | `<\|im_start\|>`, `<\|im_end\|>` |
| Phi-3 | `<\|user\|>...<\|assistant\|>` | Various special tokens |

**Critical:** Use the correct chat template for your base model. Mismatched templates cause training failures or degraded output quality. Most training frameworks (Axolotl, TRL) handle this automatically if configured correctly.

## Collection Strategies

### Manual Curation

Subject matter experts write or review every example.

**Pros:** Highest quality. Full control over distribution. Catches edge cases.
**Cons:** Expensive ($5-50 per example). Slow (10-50 examples per person per day). Doesn't scale.
**Best for:** Initial dataset (first 200-500 examples), evaluation sets, high-stakes domains (medical, legal).

### Seed + Iterate

Start with a small manually curated dataset, fine-tune, evaluate the model's outputs, and use the best outputs (after human review) as new training examples.

**Pros:** Scales better than pure manual curation. Focuses effort on cases the model gets wrong.
**Cons:** Risk of model self-reinforcement (amplifying its own biases). Requires careful human review.
**Best for:** Growing a dataset iteratively after initial fine-tuning.

### Synthetic Data Generation

Use a larger model (GPT-4, Claude) to generate training examples for a smaller model.

**Pros:** Fast and scalable. Can generate thousands of examples in hours.
**Cons:** Quality ceiling limited by the generating model. Risk of style homogenization. May violate terms of service (check provider policies). Can amplify the generating model's biases and errors.
**Best for:** Bootstrapping a dataset when no examples exist. Augmenting a small human-curated dataset.

### Distillation

Train a smaller model to mimic a larger model's behavior on your specific task.

**Pros:** Effective at transferring task-specific capabilities. Well-studied approach.
**Cons:** Legal and licensing considerations (some model licenses prohibit distillation). The student model inherits the teacher model's mistakes.
**Best for:** Cost reduction -- replacing expensive API calls with a fine-tuned smaller model.

### Real-World Collection

Collect examples from actual usage of your application (with user consent and privacy protections).

**Pros:** Perfectly representative of real-world distribution. Captures edge cases you wouldn't think to create.
**Cons:** Requires existing deployment. Privacy and consent considerations. Raw data needs significant cleaning.
**Best for:** Improving an already-deployed model based on actual usage patterns.

## Cleaning and Preprocessing

### Deduplication

Duplicate examples waste training compute and bias the model toward duplicated content.

**Exact deduplication:** Remove identical examples. Simple string matching or hash comparison.

**Near-deduplication:** Remove examples that are substantially similar. Use MinHash, SimHash, or embedding similarity with a threshold (e.g., cosine similarity > 0.95).

**Semantic deduplication:** Remove examples that convey the same meaning in different words. More aggressive but prevents the model from over-indexing on specific topics.

### Quality Filtering

- **Length filtering** -- Remove examples with extremely short responses (likely incomplete) or extremely long responses (likely verbose/off-topic)
- **Language detection** -- Remove examples in unintended languages
- **Toxicity filtering** -- Remove or flag examples containing harmful content
- **Format validation** -- Ensure all examples match the expected schema (valid JSON, correct role labels, etc.)
- **Perplexity filtering** -- Use a reference model to score examples; remove those with very high perplexity (likely noise or encoding errors)

### Data Augmentation

- **Paraphrasing** -- Generate alternative phrasings of instructions to improve robustness
- **Back-translation** -- Translate to another language and back to create natural variations
- **Difficulty variation** -- Create easier and harder versions of existing examples
- **Negative examples** -- Include examples of incorrect outputs labeled as such (for preference learning / DPO)

### Handling Imbalanced Distributions

If some categories or topics are overrepresented:
- **Undersample** overrepresented categories to match the minority
- **Oversample** (with augmentation) underrepresented categories
- **Use weighted loss** during training to give more importance to rare examples
- **Ensure the training distribution matches your production distribution** -- artificial balance isn't always better

## Common Pitfalls

### Contradictory Examples
If your dataset contains examples where the same input produces different outputs, the model learns to be uncertain about the correct response. This manifests as inconsistent or hedging outputs.

**Fix:** Review pairs of similar instructions and ensure their responses are consistent.

### Distribution Mismatch
Training on academic or formal examples but deploying for casual conversation (or vice versa) produces awkward outputs.

**Fix:** Ensure training examples mirror real-world usage in vocabulary, tone, and complexity.

### Contaminated Evaluation
Using similar examples in both training and evaluation sets gives falsely optimistic results.

**Fix:** Split by topic or time, not randomly. Ensure no near-duplicates exist across splits.

### Over-Reliance on Synthetic Data
Datasets generated entirely by LLMs tend to be homogeneous in style and can reinforce the generating model's biases.

**Fix:** Mix synthetic data with human-curated examples. Always include a human-reviewed evaluation set.

### Ignoring System Prompt Consistency
If your training examples use different system prompts (or no system prompt), the model may behave unpredictably at inference time.

**Fix:** Use a consistent system prompt across all training examples that matches what you will use in production.

### Label Leakage in Classification Tasks
Including the answer in the instruction ("Classify this positive review...") teaches the model to look for cues in the prompt rather than analyzing the content.

**Fix:** Ensure instructions don't contain the answer. Use neutral phrasing.

## Tools and Infrastructure

### Annotation and Curation

| Tool | Type | Best For |
|---|---|---|
| Argilla | Open-source annotation | Dataset review, labeling, quality control |
| Label Studio | Open-source annotation | Multi-modal annotation, complex labeling tasks |
| Prodigy | Commercial (SpaCy) | Fast annotation with active learning |
| Scale AI | Commercial platform | Large-scale labeling with human annotators |
| Humanloop | Commercial platform | Prompt experimentation + data collection |

### Data Processing

| Tool | Purpose |
|---|---|
| Hugging Face Datasets | Loading, processing, and sharing datasets |
| datatrove | Large-scale text data processing and deduplication |
| text-dedup | Near-deduplication using MinHash, SimHash |
| cleanlab | Automated data quality analysis and issue detection |

### Dataset Hosting

| Platform | Features |
|---|---|
| Hugging Face Hub | Version control, dataset cards, community sharing |
| Weights & Biases | Artifact tracking, versioning, lineage |
| DVC (Data Version Control) | Git-like versioning for datasets |

## Key Takeaways

1. **Quality beats quantity, always.** 500 carefully curated examples consistently outperform 50,000 noisy examples. Invest in review and curation before scaling collection.

2. **Match your training distribution to production.** The most common failure mode is training on data that looks different from what the model will encounter in the real world.

3. **Use the correct format and template.** Instruction-response for single-turn, chat format for multi-turn. Always use the chat template that matches your base model.

4. **Deduplicate aggressively.** Exact, near, and semantic deduplication prevent the model from over-indexing on repeated content.

5. **Mix collection strategies.** Start with manual curation for quality, augment with synthetic data for scale, and refine with real-world examples once deployed.

6. **Invest in annotation tooling.** Tools like Argilla and Label Studio pay for themselves quickly by making review and quality control efficient.

7. **Never contaminate your evaluation set.** Keep training and evaluation data strictly separated by topic or time to get honest performance estimates.

8. **Version your datasets.** Track changes to training data as carefully as you track code changes. Dataset regressions are harder to debug than code bugs.

## References

### Data Quality
1. Zhou, C., Liu, P., Xu, P., et al. (2023). "LIMA: Less Is More for Alignment" -- Demonstrating that 1,000 quality examples match 52,000+ noisy ones
2. Touvron, H., Martin, L., Stone, K., et al. (2023). "Llama 2: Open Foundation and Fine-Tuned Chat Models" -- Data curation methodology for instruction tuning

### Synthetic Data
3. Wang, Y., Kordi, Y., Mishra, S., et al. (2023). "Self-Instruct: Aligning Language Models with Self-Generated Instructions" -- Framework for LLM-generated training data
4. Taori, R., Gulrajani, I., Zhang, T., et al. (2023). "Stanford Alpaca: An Instruction-following LLaMA Model" -- Using GPT-3.5 to generate 52K instruction-following examples
5. Xu, C., Sun, Q., Zheng, K., et al. (2024). "WizardLM: Empowering Large Language Models to Follow Complex Instructions" -- Evol-Instruct for generating progressively harder examples

### Data Processing and Quality
6. Penedo, G., Malartic, Q., Hesslow, D., et al. (2023). "The RefinedWeb Dataset for Falcon LLM" -- Large-scale data cleaning and deduplication methodology
7. Lee, K., Ippolito, D., Nystrom, A., et al. (2022). "Deduplicating Training Data Makes Language Models Better" -- Quantifying the impact of deduplication on model quality

### Tools and Frameworks
8. Argilla (2024). "Argilla Documentation" -- Open-source data curation and annotation platform
9. Label Studio (2024). "Label Studio Documentation" -- Multi-modal data annotation tool
10. Hugging Face (2024). "Datasets Library Documentation" -- Standard library for loading and processing ML datasets
