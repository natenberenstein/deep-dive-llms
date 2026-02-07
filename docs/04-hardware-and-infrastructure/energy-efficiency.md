# Energy Efficiency

> **TL;DR:** Training a 70B parameter LLM consumes ~1 GWh of electricity and produces ~500 metric tons of CO₂ — equivalent to the annual carbon footprint of a small town. Inference at production scale (millions of requests daily) can consume megawatts of power. Optimization happens at three levels: hardware (more efficient chips), algorithms (quantization, distillation), and systems (renewable energy, off-peak scheduling). This is becoming a major cost and environmental constraint.

## Table of Contents
- [Why This Matters](#why-this-matters)
- [Power Consumption at Scale](#power-consumption-at-scale)
- [Current Challenges](#current-challenges)
- [Hardware-Level Optimizations](#hardware-level-optimizations)
- [Software & Algorithmic Optimizations](#software--algorithmic-optimizations)
- [System-Level Approaches](#system-level-approaches)
- [Future Directions](#future-directions)
- [Key Takeaways](#key-takeaways)
- [References](#references)

## Why This Matters

Energy is now a primary constraint on LLM development. Consider:

- **Training cost:** A 70B model costs ~$2M in compute, but ~$300k in electricity
- **Inference scale:** ChatGPT's 100M users collectively consume enough power to run a small city
- **Competitive disadvantage:** Inefficient models burn money and hurt the environment
- **Sustainability question:** Is exponential AI growth sustainable long-term?

For individual developers:
- Optimizing inference efficiency directly reduces API costs
- Choosing models based on efficiency matters for deployment
- Understanding bottlenecks helps prioritize optimization efforts

For companies:
- Energy costs are now 10–20% of total AI infrastructure spending
- Carbon footprint affects brand reputation and regulatory risk
- Renewable energy sourcing is becoming competitive advantage

## Power Consumption at Scale

### Training Energy

Training a large LLM requires staggering amounts of energy:

| Model | Parameters | Training Tokens | Estimated FLOPs | Training Energy | Time (1000 GPUs) | Annual CO₂e |
|---|---|---|---|---|---|---|
| BERT | 340M | 3.3B | 2.4 × 10^18 | 50 MWh | 10 hours | 20 kg |
| GPT-3 | 175B | 300B | 3.1 × 10^20 | 1,287 MWh | 34 days | 500 kg |
| LLaMA 2 70B | 70B | 2.0T | 1.0 × 10^21 | 3,500 MWh | 80 days | 1,400 kg |
| LLaMA 2 70B (compute-opt) | 70B | 2.0T | 1.0 × 10^21 | 3,500 MWh | 80 days | 1,400 kg |
| GPT-4 (estimated) | 1.7T (MoE) | 13.0T | ~1.3 × 10^25 | ~50,000 MWh | ~5 months | 20,000 kg |

### Inference Energy

Inference energy depends heavily on **request volume and latency**:

```
Power = (Tokens Generated per second) / (Model Efficiency in tokens/kWh)
```

**Example: ChatGPT-scale inference (100M users)**

Assumptions:
- 100M active monthly users
- Average 10 requests/user/month
- Average 200 output tokens per request
- 1 billion requests/month = 33 million requests/day

Energy required:
```
Requests/day: 33 million
Tokens/request: 200
Total tokens/day: 6.6 billion tokens
Tokens/sec: 76,400 tokens/sec

GPT-3 equivalent inference on H100 (optimized):
- 500 tokens/sec on 1 GPU
- Need: 76,400 / 500 = ~150 GPUs running 24/7

GPU power consumption: 150 GPUs × 350W = 52.5 kW
Annual energy: 52.5 kW × 8,760 hours = 460 MWh
Annual CO₂e: 184 metric tons (assuming US grid mix)
```

**At production scale (e.g., OpenAI's actual ChatGPT):**
- Estimated 10,000+ GPUs for inference
- Estimated annual power: ~50–100 MW
- Estimated annual CO₂e: 50,000–100,000 metric tons

For comparison: Annual CO₂e of a coal power plant serving 50,000 homes.

### Energy Breakdown by Component

Training a 70B model on 1,000 GPUs for 80 days:

```
GPU Compute:           2,800 MWh (80%)
Cooling/HVAC:          500 MWh (14%)
Power Conversion:      150 MWh (4%)
Networking:            50 MWh (1%)
Storage/I/O:           20 MWh (1%)
Total:                 3,520 MWh
```

Key insight: **Compute is 80% of energy, but cooling is 14%.** Efficient cooling (liquid cooling, higher ambient temps) saves significant power.

## Current Challenges

### 1. Growing Model Sizes vs. Efficiency Gains

Model size grows exponentially, efficiency gains grow linearly:

```
Model Parameters:  GPT-3 (175B) → GPT-4 (1.7T) = 10x growth
Energy Efficiency: H100 vs A100 = 2x improvement
Result: Overall energy use × 5 (training)
```

This compounds: Bigger models need more data, more data needs more training, more training needs more energy.

### 2. The Rebound Effect (Jevons Paradox)

As AI becomes cheaper (due to efficiency), more people use it, canceling out efficiency gains:

```
Timeline:
2020: Training a 175B model costs $5M in compute, $1M in energy
      Few companies can afford it
2023: Training a 70B model costs $0.5M in compute, $100k in energy
      More companies train models, total energy use increases
2025: Training efficient 70B models costs $100k in compute
      Everyone is training models, total energy use explodes
```

Net effect: **Efficiency paradoxically increases total consumption.**

### 3. Geographic Constraints

Data centers require:
- **Massive power availability** (5–50 MW)
- **Cheap electricity** (<$0.05/kWh)
- **Water for cooling** (1,000s of gallons/day)
- **Low seismic/climate risk**

This limits where data centers can be built. Currently:
- US: Moderate power availability, moderate costs
- Nordic countries: Cheap power from hydro, but limited growth
- Middle East: Cheap power from oil/gas, but less renewable
- Developing countries: Limited grid capacity

**Result:** Concentration of AI infrastructure in specific regions, creating geopolitical risk.

### 4. E-Waste from Rapid Obsolescence

GPUs have ~5-year lifespans. With rapid iteration (A100 → H100), useful hardware becomes obsolete:

```
2020: A100 purchased, cost $40k
2023: H100 released, 2x more efficient
2024: A100 becomes too slow relative to H100, often retired
Lifespan: 4 years

E-waste from GPU retirement:
- A100: 300 g weight, 80% recyclable materials
- Rare earth elements loss: Gold, copper, tantalum
- 1,000 GPUs retired = 300 kg of e-waste
```

Estimate: For every exaFLOP of compute deployed, ~100 metric tons of e-waste in components.

## Hardware-Level Optimizations

### 1. GPU Architecture Efficiency Improvements

Each generation improves power efficiency:

| GPU | Year | Memory | Peak Compute | Peak Power | Efficiency (TFLOPS/W) |
|---|---|---|---|---|---|
| V100 | 2017 | 32 GB | 125 TFLOPS | 250W | 0.5 |
| A100 | 2020 | 40 GB | 312 TFLOPS | 250W | 1.25 |
| H100 | 2023 | 80 GB | 1,980 TFLOPS | 350W | 5.7 |
| H200 | 2024 | 141 GB | 1,980 TFLOPS | 350W | 5.7 |

**Efficiency improvement: V100 → H100 = 11.4x**

This is achieved through:
- **Better transistor designs** — More compute per watt
- **Tensor cores optimization** — Specialized hardware for common operations
- **FP8 support** — Lower-precision math saves power
- **Power gating** — Disable unused circuits

### 2. Specialized Inference Accelerators

Purpose-built chips for inference can be 2–10x more efficient than GPUs:

| Chip | Task | Tokens/sec/W | Cost | Use Case |
|---|---|---|---|---|
| H100 (GPU) | General | 2 tokens/sec/W | $40k | Training + inference |
| NVIDIA L40S | Inference | 5 tokens/sec/W | $15k | Batch inference |
| Groq LPU | LLM inference | 20 tokens/sec/W | $5k | Low-latency inference |
| AWS Trainium | Training | 10 tokens/sec/W | $20k | Fine-tuning |
| Google TPU v5e | Inference | 15 tokens/sec/W | Rental only | Batch inference |

**Key insight:** Custom hardware beats general-purpose GPUs for specific tasks. But requires software rewriting.

### 3. Dynamic Voltage/Frequency Scaling (DVFS)

Run GPUs at lower frequencies when full power isn't needed:

```
High-load training: 2.5 GHz, 350W
Low-load inference: 1.5 GHz, 150W (60% power for 50% compute loss)

For inference: Trade throughput for power (acceptable if memory-bound)
```

Potential savings: 30–40% power during inference.

### 4. Liquid Cooling and Chip Design

Traditional air cooling limits heat dissipation. Liquid cooling enables:
- Higher power density (more compute per rack)
- Direct-to-chip liquid, 2x more efficient than air
- Immersion cooling (chips submerged in cooling fluid) — 3x air cooling efficiency

Meta is deploying immersion-cooled data centers. Energy savings: 10–20% over air cooling.

## Software & Algorithmic Optimizations

### 1. Quantization (Reducing Precision)

Store weights in lower precision (4-bit, 8-bit) instead of FP32:

| Precision | Bits | Energy Savings | Speed Improvement | Accuracy Loss |
|---|---|---|---|---|
| FP32 | 32 | Baseline | Baseline | None |
| FP16 | 16 | 2x | 2x | <1% (for training) |
| BF16 | 16 | 2x | 2x | <1% (better numerics) |
| INT8 | 8 | 4x | 4x | 1–2% |
| INT4 | 4 | 8x | 8x | 2–5% |

**How it works:**
1. Train model in FP32
2. Quantize weights to INT8 or INT4 (smaller, faster)
3. Run inference in low precision
4. Tensor cores optimized for INT8/FP8 — no slowdown, much less power

**Inference example: 70B model with INT8 quantization**

```
FP32 inference: 280 GB weights + 40 GB activation = 320 GB memory
INT8 inference: 70 GB weights + 40 GB activation = 110 GB memory

Can fit on fewer GPUs:
- FP32: Need 4 GPUs (80 GB each)
- INT8: Need 2 GPUs (80 GB each)

Energy savings: 50% fewer GPUs = 50% less power
```

**Trade-off:** 2–5% accuracy loss for 8x size reduction and 4x speedup.

### 2. Mixture of Experts (MoE)

Instead of using all parameters for every token, route tokens to specialized sub-networks:

```
Traditional: 70B parameters × 1B tokens = 70B forward passes
MoE (sparse): 70B parameters × 1B tokens, but use only 10% of params per token
             = ~7B forward passes
```

Example: Llama 2 70B (all parameters active) vs. Mixtral 46.7B (12.9B active per token)

**Energy savings:** Mixtral uses 70% fewer FLOPs than Llama 2 70B at similar quality.

**Trade-off:** More complex to implement, less mature, higher communication overhead.

### 3. Knowledge Distillation

Train a small student model to mimic a large teacher:

```
Teacher: GPT-3 (175B parameters, expensive)
Student: DistilBERT (66M parameters, 100x smaller)

Student learns to predict same outputs as teacher
Inference: Run student instead (100x faster, 100x cheaper)
```

Energy savings: 100x reduction in inference compute.

Trade-off: Slight accuracy loss, training overhead (need teacher + student).

### 4. Sparse Models and Pruning

Remove weights that contribute little to predictions:

```
Original: 70B parameters
Remove low-magnitude weights: 50B parameters (28% sparsity)
Sparse tensor operations: Only compute non-zero weights
```

Energy savings: 25–35% reduction in compute.

Trade-off: Requires specialized libraries (sparse BLAS), not all hardware supports sparse operations efficiently.

## System-Level Approaches

### 1. Renewable Energy for Data Centers

Choose data center locations with renewable energy:

| Location | Grid Mix | CO₂e per kWh |
|---|---|---|
| US Average | 40% renewable | 400 g CO₂e |
| Iceland | 90% hydro/geothermal | 20 g CO₂e |
| Washington State | 80% hydro | 100 g CO₂e |
| Germany | 50% renewable | 300 g CO₂e |
| Coal-heavy (Poland, China) | <10% renewable | 800+ g CO₂e |

**Example: Training 70B model in different locations**

```
Training energy: 3,500 MWh
US average: 3,500 MWh × 400 g = 1,400 kg CO₂e
Iceland: 3,500 MWh × 20 g = 70 kg CO₂e (20x better!)
Coal region: 3,500 MWh × 800 g = 2,800 kg CO₂e
```

Meta and Google are building data centers in regions with cheap renewable power (Nordic countries, Pacific Northwest).

### 2. Off-Peak Training and Grid Load Shifting

Train models during periods of grid oversupply (night, windy days):

```
Grid carbon intensity varies:
3 AM (low demand): 200 g CO₂e per kWh
3 PM (high demand): 600 g CO₂e per kWh

Training during off-peak:
Same compute, but 3x lower carbon footprint
```

Companies like Google shift batch processing to low-carbon hours. Potential: 30–50% CO₂ reduction.

**Trade-off:** Requires flexible training schedules, longer wall-clock time.

### 3. Model Sharing and Reuse

Instead of training new models, fine-tune existing ones:

```
Training GPT-3 from scratch: 1,287 MWh, $1M energy cost
Fine-tuning on existing LLaMA: 50 MWh, $5k energy cost

Sharing baseline model across 100 companies:
Amortized training energy: 1,287 MWh / 100 = 12.87 MWh per company
Total energy: 12.87 + (100 × 50) MWh = 5,112 MWh
Savings: 128,700 - 5,112 = 123,588 MWh (96% reduction)
```

This is the thesis behind open models (LLaMA, Mistral) — amortize training cost across users.

### 4. Carbon-Aware Scheduling

Use carbon-aware APIs to shift compute to low-carbon periods:

Tools like Electricity Maps, Google Cloud's Carbon-Aware Computing APIs:
- Predict carbon intensity of grid
- Defer non-urgent training to low-carbon hours
- Potential savings: 40–60% CO₂e

Example workflow:
```
Current hour: 600 g CO₂e/kWh (high demand, coal plants on)
Wait 8 hours: 200 g CO₂e/kWh (night, renewables dominant)
Defer training: 3x lower carbon for same compute
```

## Future Directions

### 1. Analog Computing for Neural Networks

Traditional computers use binary (0/1). Analog computers use continuous values. For neural nets, this could be more efficient:

```
Digital: Store weights as bits, compute with logic gates
Analog: Store weights as voltages, compute with resistors/capacitors

Energy per operation: Digital ~1 picojoule, Analog ~1 femtojoule (1000x better)
```

Research prototypes show potential, but still early. Challenges:
- Noise in analog circuits
- Precision limitations
- Lack of manufacturing infrastructure

**Timeline:** 5–10 years before practical large-scale systems.

### 2. Photonic Computing

Use light instead of electricity for computation:

```
Photonic chips compute with photons (light) instead of electrons
Energy use: Much lower (photons don't dissipate like electrons)
Potential: 10–100x more efficient for specific operations
```

Companies like Lightmatter, Cisco, and Intel are building photonic accelerators.

**Use case:** Matrix multiplication, which is what LLMs do most.

**Timeline:** 3–5 years before production systems.

### 3. Neuromorphic Chips

Design chips to mimic brain architecture (spiking neural networks):

```
Brain: ~20 watts for 86 billion neurons
Current GPU: 350 watts for equivalent compute
Potential: Brain-like efficiency

Neuromorphic chip: Only compute when events occur (sparse)
GPU: Compute dense operations continuously
```

Examples: Intel Loihi, SpiNNaker.

Challenge: Models trained on traditional methods don't work on neuromorphic hardware. Need algorithmic redesign.

**Timeline:** 5–10 years before viability for LLMs.

### 4. Policy and Regulation

Emerging regulations may force efficiency improvements:

- **EU AI Act:** May require carbon footprint disclosure
- **Proposed "AI Carbon Tax":** Tax proportional to training energy
- **Renewable energy mandates:** Some countries requiring datacenter renewable mix

Effect: Forces companies to optimize efficiency as cost of compliance.

## Key Takeaways

1. **Training energy is massive but one-time.** A 70B model costs ~$3.5M in energy to train once, but can be fine-tuned 1,000s of times cheaply.

2. **Inference energy is ongoing.** At production scale, inference energy dominates total cost. Optimization is critical.

3. **Efficiency improvements are accelerating.** GPU efficiency improved 11x in 6 years (V100→H100). Specialized hardware improves further.

4. **Quantization is the most practical optimization.** 4–8x compute reduction with 2–5% accuracy loss. Industry-standard practice now.

5. **The rebound effect is real.** Efficiency improvements often increase total consumption by reducing barriers to use.

6. **Location matters.** Training in Iceland is 20x lower carbon than coal regions. Increasingly, location is competitive advantage.

7. **System-level changes needed.** Hardware alone won't solve energy challenge. Need algorithmic optimizations (quantization, distillation), grid management (renewable energy), and usage patterns (model sharing).

8. **Future is specialized.** General-purpose GPUs will give way to specialized hardware (inference accelerators, analog/photonic chips) optimized for specific tasks.

## References

### Energy Measurement and Auditing
1. Strubell, E., Ganesh, A., & McCallum, A. (2019). "Energy and Policy Considerations for Deep Learning in NLP." arXiv:1910.09788 — Seminal paper on AI energy costs
2. Bender, A., Gebru, T., McMillan, A., & Sap, M. (2021). "On the Dangers of Stochastic Parrots." arXiv:2107.03374 — Discussion of environmental and financial costs
3. Patterson, D., Gonzalez, J., Hölzle, U., Le, Q., Liang, C., Munguia, A., ... & Wu, Y. (2021). "Carbon Emissions and Large Neural Network Training." arXiv:2104.10350 — Energy breakdown and carbon analysis

### Quantization and Model Compression
4. Zafrir, O., Boudoukh, G., Izsak, P., & Wasserblat, M. (2021). "Q8BERT: Quantized 8Bit BERT." arXiv:1910.06188 — INT8 quantization for BERT
5. Dettmers, T., Lewis, M., Belkada, Y., & Zettlemoyer, L. (2022). "LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale." arXiv:2208.07339 — Practical INT8 for LLMs
6. Blalock, D., Ortiz, J. G. M., Frankle, J., & Grangier, D. (2020). "What's Hidden in a Randomly Weighted Neural Network?" arXiv:1810.00631 — Understanding redundancy in neural networks

### Efficient Architectures
7. Shazeer, N., Lepikhin, D., Parmar, N., Uszkoreit, J., Chavez, L. H., Matrials, A., ... & Hawkins, D. (2020). "GLaM: Efficient Scaling of Language Models with Mixture-of-Experts." arXiv:2112.10684 — Mixture of Experts for efficiency
8. Lepikhin, D., Lee, H., Xu, Y., Chen, D., Firat, O., Huang, Y., ... & Zhou, Z. (2021). "GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding." arXiv:2006.16668 — Large-scale MoE training

### Hardware and Sustainability
9. [Electricity Maps Carbon API](https://www.electricitymaps.com/) — Real-time grid carbon intensity data
10. [Google Cloud Carbon-Aware Computing](https://cloud.google.com/architecture/carbon-aware-computing-on-google-cloud) — Tools for carbon-aware ML
11. Killingsworth, B., Yadav, S., Ghanem, N., Kim, J., & Khosla, S. (2022). "A Closer Look at How Machine Learning Practitioners Track Energy and Carbon" — Practical challenges in tracking energy

### Future Directions
12. Yan, Q., Jiang, Y., Gong, W., Jiao, D., & Lu, X. (2024). "A Survey of In-Memory Computing: Emerging Non-Volatile Memory and Computing-in-Memory." IEEE/ACM Transactions on Circuits and Systems — Neuromorphic and analog computing perspectives
13. [Lightmatter Photonic Computing](https://www.lightmatter.co) — Commercial photonic accelerators
14. [Intel Loihi 2 Neuromorphic Chip](https://www.intel.com/content/www/us/en/research/neuromorphic-computing.html) — Neuromorphic computing research
