# Distributed Training Parallelism: A Complete Tutorial

This document explains how large models are trained across multiple GPUs from first principles.
It is written for someone with general programming and ML knowledge but no prior experience
with distributed training.

---

## Part 1: Foundations — What Happens in One Training Step

### 1.1 The Three Phases of Training

Every training step has three phases:

**Forward pass.** Feed a batch of inputs through the model layer by layer. Each layer reads
its input, applies a weight matrix multiplication and activation function, and passes the result
to the next layer. The output of the final layer is a prediction. A loss function compares the
prediction to the ground truth labels and produces a single scalar number — the loss — measuring
how wrong the prediction was.

**Backward pass.** Starting from the loss, work backwards through each layer using the chain
rule of calculus. Consider a single linear layer `y = W × x` where `W` is the weight matrix
and `x` is the input from the previous layer. During backward, this layer receives from the
layer above it the gradient of the loss with respect to its *output*: `∂L/∂y`. Using the chain
rule it computes two distinct things:

```
∂L/∂W = ∂L/∂y × x^T     ← gradient wrt the layer's parameters  (same shape as W)
∂L/∂x = W^T  × ∂L/∂y    ← gradient wrt the layer's input       (same shape as x)
```

`∂L/∂W` is accumulated into `W.grad` and later used by the optimizer to update the weights —
this is what the table in 1.2 refers to as "gradients." `∂L/∂x` is passed backward to the
previous layer as *its* upstream gradient `∂L/∂y`, and then discarded — it is never stored.
So each layer only keeps one gradient tensor (`∂L/∂W`), not two.

Notice that computing `∂L/∂W` requires `x` — the layer's forward-pass input. This is why
activations must be kept in memory after the forward pass: the backward pass needs them to
compute parameter gradients. This is the direct reason gradient checkpointing saves memory
(explained below).

**Optimizer step.** Use the gradients to update the weights. For Adam/AdamW, each parameter
`w` is updated using: `w = w - lr × gradient / (sqrt(variance_estimate) + epsilon)`. The
optimizer maintains two extra per-parameter running averages — the mean of recent gradients and
the mean of recent squared gradients — to estimate this adaptive learning rate.

### 1.2 What Lives in GPU Memory

For a model with P parameters (each stored as a 16-bit brain-float = 2 bytes):

| What | Size per parameter | Total for 4B model |
|---|---|---|
| Parameters | 2 bytes (bfloat16) | 8 GB |
| Gradients | 2 bytes (bfloat16) | 8 GB |
| Adam first moment (mean of grads) | 4 bytes (float32) | 16 GB |
| Adam second moment (variance of grads) | 4 bytes (float32) | 16 GB |
| **Total model state** | **12 bytes** | **48 GB** |

The first and second moments are stored in float32 (not bfloat16) because the small incremental
updates to these moments would lose precision in 16-bit arithmetic, causing training instability.

**Activations** are the intermediate outputs stored during the forward pass so the backward pass
can use them. For a Transformer with `L` layers, sequence length `S`, and hidden dimension `D`,
storing all activations takes roughly `O(L × S × D)` memory. For long-context training
(64k, 128k tokens), activations become the dominant memory cost — not the model weights.

**Attention is O(S²).** In the attention mechanism, each token attends to every other token.
The naive attention weight matrix has shape `[batch_size, heads, S, S]` — one scalar for every
(query token, key token) pair, for every head, for every sequence in the batch. For S=65536
(64k) with a batch size of 1 and 8 heads, that is already 8 × 65536 × 65536 ≈ 34 billion
entries. At 2 bytes each, ~68 GB — clearly infeasible on a single GPU.

In practice, **FlashAttention** avoids materializing this matrix entirely. It computes
attention in small tiles that fit in the GPU's fast on-chip SRAM, streaming through the
sequence in blocks and accumulating the result without ever writing the full `[S, S]` matrix
per head to main GPU memory (HBM). The output is identical; peak attention *memory* drops from
O(S²) to O(S). The O(S²) cost becomes purely a *compute* cost, not a memory cost.

This is why FlashAttention is a hard requirement for long-context training, not an optional
optimization. Even with FlashAttention,
the *other* activations (layer inputs stored for the backward pass) still grow with S, which
is why sequence parallelism is also needed at very long contexts.

**Gradient checkpointing** trades compute for activation memory. Instead of storing all
activations during the forward pass, discard them. When the backward pass needs a layer's
input `x` to compute `∂L/∂W`, recompute it by replaying the forward pass from the nearest
saved checkpoint up to that layer.

The compute overhead depends critically on the replay strategy. There are three distinct cases
for an L-layer network:

**Save all activations (no checkpointing):** memory O(L), compute 1F + 2F = 3F. Baseline.
Here 1F is the forward pass and 2F is the backward pass — the backward pass costs roughly
twice a forward pass because for each layer it computes two quantities (`∂L/∂W` and `∂L/∂x`,
see section 1.1) whereas the forward pass computes only one output per layer.

**Discard everything, recompute naively:** store no activations at all — not even during
replays. To backpropagate through layer k you need `x_k` (its input). You don't have it, so
you replay forward from layer 1 to produce it. But during that replay you still store nothing
— intermediate activations are discarded as soon as the replay moves past them. So after
using `x_k` for `∂L/∂W_k`, you move to layer k-1, need `x_{k-1}` — and it's gone, having
been discarded during the previous replay. You must replay from layer 1 again. Every layer
triggers a full replay from scratch. Layer 1 is recomputed L times, layer 2 L-1 times, and
so on. Total recompute = O(L²). Memory is O(1) — only one activation live at a time — but
compute explodes quadratically. Unusable in practice.

**Checkpoint every √L layers (the standard strategy):** divide the network into √L segments
of length √L each. Store only the activation at the boundary of each segment (√L tensors
total). During backward, to backpropagate through a segment, replay that segment's √L layers
from its stored boundary — this time keeping all √L intermediate activations in memory until
the segment's backward is complete, then discarding them. Because you have the boundary
checkpoint, you never need to replay further back than the start of the current segment. Each
layer is therefore recomputed at most once across the entire backward pass. Total recompute =
1 additional forward pass. Peak memory = O(√L) stored boundary checkpoints + O(√L) live
activations during one segment's replay = O(√L) total. Compute 1F + 1F + 2F = 4F,
overhead = 33%.

The √L choice balances the two memory terms: storing more checkpoints (shorter segments)
reduces replay memory but increases stored checkpoints; fewer checkpoints (longer segments)
reduces stored checkpoints but increases the number of activations live during each replay.
√L minimises the sum.

This √L strategy is what PyTorch's `gradient_checkpointing=True` implements. The 33% figure
is specific to this strategy. Keeping activations for "half the layers" does not automatically
give you 0.5F recompute — the overhead depends on which layers you keep and how the replay
is organized, not simply on the fraction stored.

**Concrete numbers — Qwen3-4B (L=36, d=2560, GQA with 8 KV heads, FFN=9728):**

Per token per layer, the backward pass needs: layer input (d), Q (d), K and V (d/4 each, GQA),
attention output (d), FFN input (d), and FFN gate+up activations (2×FFN). In bf16 that is
~60.5 KB per token per layer.

| Strategy | 64k context | 128k context | Compute overhead |
|---|---|---|---|
| No checkpointing (store all) | **146 GB** | **292 GB** | 0% (baseline) |
| √L checkpointing (PyTorch default) | **26 GB** | **53 GB** | 33% |

The 26 GB at 64k breaks down as: 2 GB for the 6 boundary checkpoints (just the layer input
tensor at each segment boundary) + 24 GB peak during segment replay (6 layers × full
activations live at once). With 4 GPUs and SP=4, each GPU sees only S/4 tokens, reducing
these figures by 4× — 6.5 GB and 13 GB respectively. This is why gradient checkpointing is
essential at long contexts but SP is also required to bring the per-GPU footprint to a
manageable level.

---

## Part 2: Communication Primitives and Key Terms

All distributed training boils down to a handful of communication operations between GPUs.
Understanding them precisely is essential — these are the atoms of every parallelism strategy.

### 2.1 Processes, Ranks, and World Size

When you launch distributed training with `torchrun --nproc_per_node=4 train.py`, PyTorch
starts 4 independent copies of `train.py` as separate OS processes, each mapped to one GPU.

**Rank:** the integer ID assigned to a process. With 4 processes, ranks are 0, 1, 2, 3.
Rank 0 is often the "coordinator" — the process that prints logs and saves checkpoints.
Every process knows its own rank and the total number of processes. Later (section 2.2),
we will define *process groups* — subsets of processes that communicate with each other —
and each process also has a separate rank within each group it belongs to. For now, think
of rank simply as a process's global ID.

**World size:** the total number of processes launched. With `--nproc_per_node=4`, world size
= 4. This is the total GPU count available to the job.

**Local rank:** on a single machine, the per-node index of the process. If you have 2 nodes
with 4 GPUs each (world size = 8), local ranks are 0–3 on each node independently. Local rank
determines which physical GPU the process is mapped to.

### 2.2 Process Groups and Group Ranks

A *process group* is a named subset of the world's processes. When you call a communication
operation (all-reduce, all-gather, etc.) with a specific group, only the processes in that
group participate and only they can see each other's data.

Every process has an integer rank *within each group it belongs to*. These group ranks are
independent of global ranks. For example, with 4 GPUs:

```
Global ranks:  GPU0=0   GPU1=1   GPU2=2   GPU3=3

Define Group A = {GPU0, GPU1}:
  GPU0 has rank 0 in Group A
  GPU1 has rank 1 in Group A

Define Group B = {GPU2, GPU3}:
  GPU2 has rank 0 in Group B
  GPU3 has rank 1 in Group B
```

GPU0's global rank is 0, and its rank within Group A is also 0 — but if GPU2 were added to
a different group, it would be rank 0 in that group too. Group ranks reset to 0 for each group.

A single GPU can belong to multiple groups and will participate in each group's communications
at different points in the training loop:

```
Group C = {GPU0, GPU2}  (a third, overlapping group)
  GPU0 has rank 0 in Group C
  GPU2 has rank 1 in Group C
```

GPU0 belongs to Group A and Group C simultaneously. It will perform Group A's all-reduce at one
point in the step, and Group C's all-to-all at another point.

In distributed training, you define several named groups — one for data parallelism (DP group),
one for tensor parallelism (TP group), one for sequence parallelism (SP group) — and the same
GPU participates in all of them.

### 2.3 The Eight Communication Operations

All operations below happen across a process group of N members, each holding a tensor.

---

**Broadcast.** One GPU sends its tensor to all others.
```
Before:  GPU0=[A]   GPU1=[ ]   GPU2=[ ]   GPU3=[ ]
After:   GPU0=[A]   GPU1=[A]   GPU2=[A]   GPU3=[A]
```
Used for: distributing initial model weights, sharing configuration.

---

**Reduce.** Each GPU has a tensor of the same shape. They are summed element-wise. The result
lands on one designated GPU.
```
Before:  GPU0=[1,2]   GPU1=[3,4]   GPU2=[5,6]   GPU3=[7,8]
After:   GPU0=[16,20]  GPU1=[ ]    GPU2=[ ]     GPU3=[ ]
```
Used for: computing the sum of partial results at one location.

---

**All-Reduce.** Same as Reduce, but the result is delivered to *all* GPUs.
```
Before:  GPU0=[1,2]   GPU1=[3,4]   GPU2=[5,6]   GPU3=[7,8]
After:   GPU0=[16,20]  GPU1=[16,20]  GPU2=[16,20]  GPU3=[16,20]
```
Used for: standard data parallelism gradient synchronization — every GPU gets the average
gradient and can independently update its identical copy of the model.

---

**Scatter.** One GPU splits a large tensor into N equal chunks and sends chunk i to GPU i.
```
Before:  GPU0=[A,B,C,D]   GPU1=[ ]   GPU2=[ ]   GPU3=[ ]
After:   GPU0=[A]   GPU1=[B]   GPU2=[C]   GPU3=[D]
```
Used for: distributing shards of data or weights.

---

**Gather.** The inverse of Scatter. Each GPU has a chunk; they are concatenated on one GPU.
```
Before:  GPU0=[A]   GPU1=[B]   GPU2=[C]   GPU3=[D]
After:   GPU0=[A,B,C,D]   GPU1=[ ]   GPU2=[ ]   GPU3=[ ]
```

---

**All-Gather.** Same as Gather, but the concatenated result goes to all GPUs.
```
Before:  GPU0=[A]   GPU1=[B]   GPU2=[C]   GPU3=[D]
After:   GPU0=[A,B,C,D]   GPU1=[A,B,C,D]   GPU2=[A,B,C,D]   GPU3=[A,B,C,D]
```
Used for: reconstructing a full tensor from shards — critical in ZeRO3 (reassemble a layer's
full weights before computing that layer) and in Ulysses SP (reassemble the full sequence
after attention).

---

**Reduce-Scatter.** Combines a reduce and a scatter in one efficient step. Each GPU contributes
a full-size tensor. The tensors are summed element-wise, and the sum is then partitioned —
each GPU receives one contiguous shard of the total sum.
```
Before:  GPU0=[1,2,3,4]   GPU1=[1,2,3,4]   GPU2=[1,2,3,4]   GPU3=[1,2,3,4]
                (each GPU holds its local gradient contribution, same shape)
Sum:     [4,8,12,16]
After:   GPU0=[4]   GPU1=[8]   GPU2=[12]   GPU3=[16]
```
Compare this to All-Reduce: All-Reduce gives every GPU the full result `[4,8,12,16]`
(4 elements per GPU); Reduce-Scatter gives each GPU only 1 element — 1/N of the data.
The total data transferred across the network is the same in both cases, but after
Reduce-Scatter each GPU holds 1/N the memory. This is why ZeRO uses it for gradients.

---

**All-to-All.** Each GPU has N chunks and sends one different chunk to each other GPU,
simultaneously receiving one chunk from each. Think of it as transposing a 2D matrix of
data across GPUs.
```
GPU0 holds [A0, A1, A2, A3]   (chunk j goes to GPU j)
GPU1 holds [B0, B1, B2, B3]
GPU2 holds [C0, C1, C2, C3]
GPU3 holds [D0, D1, D2, D3]

After:
GPU0 = [A0, B0, C0, D0]   (column 0: chunk 0 from every GPU)
GPU1 = [A1, B1, C1, D1]   (column 1: chunk 1 from every GPU)
GPU2 = [A2, B2, C2, D2]
GPU3 = [A3, B3, C3, D3]
```
Before: each GPU holds all chunks from one source. After: each GPU holds one chunk from every
source. This is the core of Ulysses sequence parallelism — rearranging between
"token shard + all heads" and "full sequence + head shard" layouts (explained in Part 6).

---

## Part 3: Data Parallelism (DP)

### 3.1 The Core Idea

The simplest way to use multiple GPUs: give each GPU a complete copy of the model and a
different subset of the training batch. Each GPU independently runs the forward and backward
passes on its data slice. After the backward pass, all GPUs need to agree on the weight update
— so they average their gradients using an all-reduce across all of them. Every GPU then updates
its model with the same averaged gradient, so all copies stay identical. On the next step,
the process repeats with new data slices.

```
Effective batch size = per_device_batch_size × number_of_DP_GPUs

GPU0: inputs[0:B]  → forward → backward → grad_0 ──┐
GPU1: inputs[B:2B] → forward → backward → grad_1 ──┤─ all_reduce → avg_grad → each GPU updates weights
GPU2: inputs[2B:3B]→ forward → backward → grad_2 ──┤
GPU3: inputs[3B:4B]→ forward → backward → grad_3 ──┘
```

All four GPUs participate in this all-reduce together. In process group terms: **all 4 GPUs
form a single DP group of size `dp_world_size=4`**. The all-reduce above runs within this
group — only these 4 GPUs communicate, and each GPU's rank within the DP group is its global
rank (0, 1, 2, 3 for a pure DP setup).

Result: 4× faster data processing, same model quality as training on a single GPU with batch
size 4B.

Later, when we introduce sequence parallelism, ms-swift tracks an internal `dp_world_size =
total_gpus / sp_size` for data loading purposes. With SP=2 on 4 GPUs: GPU0+GPU1 collaborate
on one sequence, GPU2+GPU3 on another — so ms-swift assigns different samples to each SP pair.
`dp_world_size = 2`. However, DeepSpeed's ZeRO group remains the full world (all 4 GPUs), as
explained in detail in Part 4.5.

### 3.2 The Memory Problem

Every GPU in the DP group holds an identical, complete copy of the model: parameters,
gradients, and optimizer states. With a 4B model (48 GB of model state) and 4 GPUs, you have
4 × 48 GB = 192 GB of model state spread across GPUs — but only 48 GB is unique information.
The other 144 GB is pure duplication.

This redundancy is what ZeRO (Part 4) eliminates.

---

## Part 4: ZeRO — Eliminating Redundancy in Data Parallelism

ZeRO (Zero Redundancy Optimizer) was developed by Microsoft DeepSpeed. The key insight:
every GPU in the DP group holds identical copies of parameters, gradients, and optimizer
states. Instead of every GPU maintaining the full set, partition these across the DP group
so each GPU is responsible for 1/`dp_world_size` of the total.

**Why is this valid?** Because the all-reduce in data parallelism already aggregates everyone's
gradients. If we instead do a reduce-scatter (each GPU gets its own shard of the gradient sum),
and each GPU then updates only its assigned shard of the parameters, the resulting model is
mathematically identical to what full DP would produce — the gradients were still averaged,
just distributed rather than replicated.

**ZeRO stages are cumulative.** Each stage is a strict superset of the previous one:
- **ZeRO1**: partition optimizer states only
- **ZeRO2**: partition optimizer states *and* gradients (ZeRO1 included)
- **ZeRO3**: partition optimizer states, gradients, *and* parameters (ZeRO1 + ZeRO2 included)

You never run ZeRO2 without also getting ZeRO1's savings. In all examples below, assume a 4B
model (8 GB params in bf16, 8 GB grads in bf16, 16+16=32 GB Adam moments in fp32 = 48 GB
total) and `dp_world_size=4` (4 data-parallel replicas).

### 4.1 ZeRO Stage 1: Partition Optimizer States

**What changes:** the optimizer step. Standard DP does a full all-reduce to average gradients,
then every GPU independently runs the optimizer on all 8 GB of parameters. ZeRO1 keeps the
all-reduce but assigns each GPU responsibility for only 1/4 of the optimizer state — so each
GPU only runs the optimizer step for its assigned parameter shard.

**What stays the same:** the gradient communication. ZeRO1 still does an all-reduce (all four
GPUs exchange and sum their gradients, and every GPU ends up with the full averaged gradient
tensor). Gradient memory is unchanged.

*Standard DP — optimizer step communication:*
```
Each GPU computed gradients for all 4 parameter chunks [P0,P1,P2,P3]:

GPU0: [g0,g1,g2,g3]   ──┐
GPU1: [g0,g1,g2,g3]   ──┤── all-reduce → every GPU gets the averaged [g0,g1,g2,g3]
GPU2: [g0,g1,g2,g3]   ──┤              → every GPU runs Adam on all 4 chunks
GPU3: [g0,g1,g2,g3]   ──┘              → every GPU stores all 4 Adam moment shards
```

*ZeRO1 — same all-reduce, but only update your assigned shard:*
```
GPU0: [g0,g1,g2,g3]   ──┐
GPU1: [g0,g1,g2,g3]   ──┤── all-reduce → every GPU gets the averaged [g0,g1,g2,g3]
GPU2: [g0,g1,g2,g3]   ──┤              → GPU0 runs Adam on P0 only, stores moments for P0
GPU3: [g0,g1,g2,g3]   ──┘              → GPU1 runs Adam on P1 only, stores moments for P1
                                         GPU2 → P2, GPU3 → P3
                        then: all-gather to rebuild full updated params on all GPUs
```

Memory before and after:
```
Standard DP per GPU: 8 (params) + 8 (grads) + 32/1 (moments) = 48 GB
ZeRO1 per GPU:       8 (params) + 8 (grads) + 32/4 (moments) = 24 GB  ← half
```

### 4.2 ZeRO Stage 2: Partition Optimizer States + Gradients

ZeRO2 includes everything ZeRO1 does, and additionally changes the gradient communication.
Instead of an all-reduce (which gives every GPU the full averaged gradient), ZeRO2 does a
reduce-scatter: each GPU ends up with only the gradient shard for its assigned parameters.
The gradient memory for the other 3/4 of parameters is freed immediately.

*Standard DP (and ZeRO1) — all-reduce after backward:*
```
GPU0 local grads: [g0,g1,g2,g3]   ──┐
GPU1 local grads: [g0,g1,g2,g3]   ──┤── all-reduce
GPU2 local grads: [g0,g1,g2,g3]   ──┤
GPU3 local grads: [g0,g1,g2,g3]   ──┘
Result: every GPU holds the full averaged gradient [g0,g1,g2,g3] (8 GB)
```

*ZeRO2 — reduce-scatter instead:*
```
GPU0 local grads: [g0,g1,g2,g3]   ──┐
GPU1 local grads: [g0,g1,g2,g3]   ──┤── reduce-scatter
GPU2 local grads: [g0,g1,g2,g3]   ──┤
GPU3 local grads: [g0,g1,g2,g3]   ──┘
Result: GPU0 holds only averaged g0 (2 GB)
        GPU1 holds only averaged g1 (2 GB)
        GPU2 holds only averaged g2 (2 GB)
        GPU3 holds only averaged g3 (2 GB)
```

Each GPU then runs the Adam optimizer on its shard using its shard of gradients. No
all-gather is needed before the optimizer step (each GPU already has exactly the gradients
it needs). After the optimizer step, an all-gather reconstructs the full updated parameters
on all GPUs before the next forward pass — same as ZeRO1.

Memory:
```
ZeRO1 per GPU: 8 (params) + 8    (grads) + 8 (moments) = 24 GB
ZeRO2 per GPU: 8 (params) + 8/4  (grads) + 8 (moments) = 18 GB  ← 25% less
```

Memory saved vs. standard DP: 18 GB per GPU vs 48 GB — a 2.7× reduction.

### 4.3 ZeRO Stage 3: Partition Everything Including Parameters

ZeRO3 includes everything ZeRO1 and ZeRO2 do, and additionally partitions the parameters
themselves. At rest, each GPU holds only 1/4 of the parameters.

The problem: during forward and backward, each layer needs its full weight matrix to compute
the layer output. ZeRO3 solves this by running an all-gather just before each layer is
computed, reconstructing the full weights temporarily. After the layer computation completes,
the non-owned parameter shards are discarded.

*ZeRO3 forward pass — one transformer layer:*
```
At rest (before layer L):
  GPU0 holds params[L][shard0]   GPU1 holds params[L][shard1]
  GPU2 holds params[L][shard2]   GPU3 holds params[L][shard3]

  ── all-gather ──►  all GPUs temporarily hold full params[L]
  compute layer L output on all GPUs (each using different input slice from DP)
  ── discard shards 1,2,3 on GPU0 / shards 0,2,3 on GPU1 / etc. ──

At rest (after layer L):
  GPU0 holds params[L][shard0]   GPU1 holds params[L][shard1]   (back to sharded state)
```

*ZeRO3 backward pass — same layer L:*
```
  ── all-gather params[L] again ──►  need full weights to recompute activations + gradient
  compute ∂L/∂W and ∂L/∂x for layer L
  ── reduce-scatter ∂L/∂W ──►  each GPU gets its shard of the averaged gradient
  ── discard reconstructed params[L] (non-owned shards) ──
```

For a 4B model with 36 transformer layers: 36 all-gathers during forward + 36 all-gathers
+ 36 reduce-scatters during backward = 108 communication operations in total, vs. 1 all-gather
at the end of ZeRO1/2. This is significant communication overhead — the price of fitting the
model on fewer GPUs.

Memory:
```
ZeRO2 per GPU: 8     (params) + 8/4 (grads) + 8 (moments) = 18   GB
ZeRO3 per GPU: 8/4   (params) + 8/4 (grads) + 8 (moments) = 12   GB  ← 33% less
```

Memory saved vs. standard DP: 12 GB per GPU vs 48 GB — a 4× reduction at rest.
(During forward/backward, the temporarily reconstructed weights add up to ~8 GB again
for the layer currently being computed, so the peak is higher than 12 GB.)

### 4.4 ZeRO-Offload

An extension to any ZeRO stage: move optimizer states (and with ZeRO3, also parameters and
gradients) to CPU RAM instead of GPU memory, streaming them back to GPU as needed. CPU RAM is
much larger (hundreds of GB vs 40–80 GB per GPU) but ~10× slower than GPU HBM for large
transfers.

`zero3_offload` = ZeRO3 + CPU offload of optimizer states and gradients. This maximizes
GPU memory available for activations at the cost of slower optimizer steps (the CPU-GPU
transfer becomes the bottleneck rather than the GPU computation itself).

### 4.5 ZeRO with Sequence Parallelism — the mpu question

#### The naive intuition — and why it is incomplete

Consider 4 GPUs running with SP=4. From Part 3.1: SP forms one group of 4 GPUs that all
cooperate on the same sequence. Only *different* SP groups represent independent DP replicas.
With SP=4 on 4 GPUs there is only one SP group — no independent replicas at all.

The formula from Part 3.1:
```
dp_world_size = total_gpus / sequence_parallel_size = 4 / 4 = 1
```

A DP group of size 1 means each GPU is its own replica. ZeRO shards across the DP group —
sharding across a group of size 1 gives nothing. By this reasoning: use `zero0`, anything
above is wasted overhead.

This reasoning is correct *if* DeepSpeed knows about your SP groups. Whether it does depends
on a concept called the **mpu**.

#### What mpu is

DeepSpeed is a general optimizer library. It knows nothing about how you arranged your GPUs
into parallel groups — unless you tell it. The way you tell it is by passing an **mpu**
(Model Parallel Unit) object when initializing DeepSpeed:

```python
# With mpu: DeepSpeed knows about your parallel topology
deepspeed.initialize(model, optimizer, mpu=mpu)

# Without mpu: DeepSpeed sees nothing except a flat list of N ranks
deepspeed.initialize(model, optimizer)
```

The mpu is a Python object that exposes a small interface:
```python
mpu.get_data_parallel_group()        # which ranks are DP replicas
mpu.get_data_parallel_world_size()   # how many DP replicas
mpu.get_data_parallel_rank()         # this GPU's rank within DP group
```

When DeepSpeed receives an mpu, it calls `mpu.get_data_parallel_group()` and uses that process
group for all ZeRO gradient reductions and parameter sharding. It only shards and reduces
within the actual DP replicas — SP ranks are in a different group and are invisible to ZeRO.

#### Case 1 — with mpu (e.g. Megatron-DeepSpeed): 4 GPUs, SP=2, DP=2

The framework builds a 2D mesh before training starts:

```
Mesh layout (row = SP group, column = DP group):

        DP rank 0   DP rank 1
SP rank 0:  GPU0        GPU2
SP rank 1:  GPU1        GPU3

SP groups: {GPU0, GPU1}  and  {GPU2, GPU3}
DP groups: {GPU0, GPU2}  and  {GPU1, GPU3}
```

The mpu is constructed from this mesh:
```python
mpu.get_data_parallel_group()  # returns {GPU0,GPU2} on GPU0, {GPU1,GPU3} on GPU1, etc.
mpu.get_data_parallel_world_size()  # returns 2
```

DeepSpeed receives this mpu and uses the DP groups for ZeRO. With ZeRO-3:
- GPU0 and GPU2 shard parameters/gradients/optimizer states between themselves (dp=2, 2× savings)
- GPU1 and GPU3 do the same independently
- GPU0 and GPU1 communicate only for SP (the Ulysses all-to-all for attention heads) — ZeRO
  never touches this pair

With SP=4 on 4 GPUs in this setup:
```
Mesh: one row of 4 (SP group), one column of 1 (DP group)

DP group per GPU = {only itself} → dp_world_size = 1
ZeRO shards across a group of 1 → no sharding, no benefit
```
Here, `zero0` is the correct choice — ZeRO above stage 0 allocates communication buffers for a
group of one and achieves nothing.

#### Case 2 — without mpu (ms-swift): 4 GPUs, SP=4

ms-swift does **not** pass an mpu to DeepSpeed:

```python
deepspeed.initialize(model, optimizer)  # no mpu
```

DeepSpeed has no idea that SP groups exist. It sees 4 flat ranks and uses the entire world
process group — {GPU0, GPU1, GPU2, GPU3} — as its ZeRO reduction group. This is the same 4
GPUs that form the SP group.

ms-swift does internally compute `dp_world_size = 4 / 4 = 1` and builds a device mesh for
data loading, but this mesh is invisible to DeepSpeed. From DeepSpeed's perspective there is
one group of 4 peers and ZeRO shards across all of them:

```
DeepSpeed's view (no mpu):

  ZeRO group: {GPU0, GPU1, GPU2, GPU3}   ← all 4 SP ranks
  ZeRO-3: shards params/grads/optimizer states across all 4

ms-swift's internal view:

  SP group: {GPU0, GPU1, GPU2, GPU3}     ← same 4 GPUs, used for Ulysses all-to-all
  dp_world_size = 1                      ← used only for data loading assignment
```

The same 4 GPUs play both roles simultaneously: they are the SP group for sequence
communication, and the ZeRO group for parameter sharding.

#### Is this correct? The reduce-scatter question

With SP=4, each GPU processes tokens [0:S/4], [S/4:S/2], [S/2:3S/4], [3S/4:S] of the same
sequence. After the backward pass, each GPU holds partial gradients for *its token slice only*:

```
GPU0: ∂L/∂W from tokens 0..16k     (partial gradient)
GPU1: ∂L/∂W from tokens 16k..32k   (partial gradient)
GPU2: ∂L/∂W from tokens 32k..48k   (partial gradient)
GPU3: ∂L/∂W from tokens 48k..64k   (partial gradient)
```

The correct full gradient for weight W is the **sum** of all four. DeepSpeed's reduce-scatter
computes exactly this — it sums across all 4 ranks and distributes shards of the result:

```
After reduce-scatter (ZeRO-2/3):
  GPU0 holds: sum(GPU0..GPU3 gradients) for parameter shard 0   ← complete, not partial
  GPU1 holds: sum(GPU0..GPU3 gradients) for parameter shard 1
  GPU2 holds: sum(GPU0..GPU3 gradients) for parameter shard 2
  GPU3 holds: sum(GPU0..GPU3 gradients) for parameter shard 3
```

Reduce-scatter is not a partial sum — it is a full all-reduce whose result is distributed as
shards. Each GPU ends up with 1/4 of the *complete, fully-summed* gradient. This is the same
mathematical property that makes ZeRO-2/3 correct in ordinary DP; it holds here too.

#### The one real subtlety: gradient normalization

Standard DP averages gradients across replicas (divides by world_size=4) — the right thing
when each rank independently processed a different sample. SP ranks processed *chunks of one
sample* — the correct operation is a **sum**, not an average.

If DeepSpeed's automatic divide-by-4 is not compensated, every parameter update is 4× too
small. ms-swift handles this explicitly in `GatherLoss.backward`:

```python
# swift/sequence_parallel/utils.py
_grad = grad_output[0] * sequence_parallel.world_size
```

`grad_output[0]` arrives after DeepSpeed's reduce-scatter has already divided by `world_size=4`.
This line multiplies back by 4 to cancel that division before distributing the gradient to each
SP rank's sequence chunk. The net effect: the optimizer sees the correct summed gradient, not a
4× deflated average.

This compensation is why ms-swift uses a custom loss function (`per_token_loss_func_sp`) when
SP is active — the normalization fix lives in the backward of that custom loss, not in the
standard HuggingFace cross-entropy path.

#### Summary: mpu present vs absent

| | With mpu (Megatron-DS) | Without mpu (ms-swift) |
|---|---|---|
| ZeRO reduction group | `mpu.get_data_parallel_group()` | entire world (all GPUs) |
| SP=4, 4 GPUs — ZeRO shards? | No (dp group = size 1) | Yes (world group = 4 GPUs) |
| SP=2, 4 GPUs — ZeRO shards? | Yes (dp group = size 2) | Yes (world group = 4 GPUs) |
| Normalization handled by | framework-level DP averaging | ms-swift custom loss backward |
| `zero3_offload` + SP=4 valid? | No — useless overhead | Yes — standard configuration |

#### ZeRO-3 and frozen parameters (LoRA)

ZeRO-3 shards *all* parameters at rest, including frozen base model weights. Before every
layer during forward and backward, it runs an all-gather to reconstruct the full weight matrix
— for trainable and frozen layers alike. The optimizer step only updates trainable parameters
(DeepSpeed filters by `requires_grad`), but the all-gather overhead applies to every layer.

For Qwen3-4B with LoRA: 36 layers × (1 forward + 1 backward all-gather) = 72 all-gather
operations per step across the frozen base weights. This is significant communication overhead.
`zero3_offload` adds CPU offload of the small LoRA optimizer states on top.

---

## Part 5: Tensor Parallelism (TP)

### 5.1 The Idea

DP replicates the full model on every GPU. TP takes the opposite approach: partition the
weight matrices themselves across GPUs. Each GPU holds a different shard of each layer's
weights and computes a partial result. The partial results are combined with a communication
step. The set of GPUs sharing the same weight matrix shards is called the **TP group**.

**How TP differs from ZeRO3.** Both split weight matrices across GPUs, so they look similar
at first glance. The difference is in *why* and *when* the split is undone:

- **ZeRO3** shards weights purely to save *storage*. Before a layer is computed, it runs an
  all-gather to reconstruct the full weight matrix on every GPU. Every GPU then does the same
  complete matrix multiply on its own slice of the batch. The split is undone before any work
  happens — it only ever existed to avoid holding redundant copies in memory.

- **TP** shards weights to split the *computation itself*. The full weight matrix is never
  reconstructed on any GPU. Each GPU multiplies the full input against its weight shard and
  gets a partial result; those partial results are then combined with an all-reduce.

```
ZeRO3 — each layer:
  shards at rest → all-gather → every GPU has full W → GPU0 computes X0 @ W
                                                        GPU1 computes X1 @ W  (different data)
                                                        GPU2 computes X2 @ W
                                                        GPU3 computes X3 @ W

TP — each layer:
  GPU0 holds W_col0,  computes X @ W_col0 → partial ──┐
  GPU1 holds W_col1,  computes X @ W_col1 → partial ──┤── all-reduce → full output Y
  GPU2 holds W_col2,  computes X @ W_col2 → partial ──┤   (same data X on all GPUs)
  GPU3 holds W_col3,  computes X @ W_col3 → partial ──┘
```

ZeRO3 is about *not storing* redundant copies — computation is still replicated across GPUs.
TP is about *splitting the computation* — each GPU does less arithmetic and never sees the
full weight matrix at all.

### 5.2 Splitting a Matrix Multiply

Consider a linear layer: `Y = X @ W` where:
- `X` has shape `[batch, seq, 1024]` (input, same on all TP GPUs)
- `W` has shape `[1024, 4096]` (weight matrix, split across TP GPUs)
- `Y` has shape `[batch, seq, 4096]` (output)

With `tp_world_size=4` (4 GPUs in the TP group), each GPU holds a different shard of `W`:

**Column Parallel (split W along its output dimension, i.e. columns):**
```
GPU0: W_0 = W[:, 0:1024]    shape [1024, 1024]
GPU1: W_1 = W[:, 1024:2048] shape [1024, 1024]
GPU2: W_2 = W[:, 2048:3072] shape [1024, 1024]
GPU3: W_3 = W[:, 3072:4096] shape [1024, 1024]
```
Each GPU computes its output shard independently using the same full input X:
```
GPU0: Y_0 = X @ W_0   shape [batch, seq, 1024]
GPU1: Y_1 = X @ W_1   shape [batch, seq, 1024]
GPU2: Y_2 = X @ W_2   shape [batch, seq, 1024]
GPU3: Y_3 = X @ W_3   shape [batch, seq, 1024]
```
No cross-GPU communication is needed during this computation. An all-gather at the end
assembles full Y = [Y_0, Y_1, Y_2, Y_3] of shape [batch, seq, 4096]. Each GPU contributed
a 1024-wide slice of the final output.

**Row Parallel (split W along its input dimension, i.e. rows):**
Each GPU holds `W_i = W[i*256:(i+1)*256, :]` of shape `[256, 4096]`, and needs the
corresponding slice of the input: `X_i = X[:, :, i*256:(i+1)*256]` of shape `[batch, seq, 256]`.

```
GPU0: partial_0 = X[:,  :,   0:256] @ W[0:256,   :] = shape [batch, seq, 4096]
GPU1: partial_1 = X[:, :, 256:512]  @ W[256:512,  :] = shape [batch, seq, 4096]
GPU2: partial_2 = X[:, :, 512:768]  @ W[512:768,  :] = shape [batch, seq, 4096]
GPU3: partial_3 = X[:, :, 768:1024] @ W[768:1024, :] = shape [batch, seq, 4096]
```
Each GPU's result is a *partial sum* — the true output is
`Y = partial_0 + partial_1 + partial_2 + partial_3`. An all-reduce within the TP group
sums these and delivers the full Y to every GPU.

**In a Transformer block the two patterns are combined:**

*Attention sub-block:*

The Q, K, V projection matrices each have shape `[d_model, d_model]`. Split them column-wise:
each GPU holds the columns corresponding to its assigned attention heads. Every GPU has the
same full input `X`, multiplies it against its head columns, and gets Q/K/V tensors for its
head subset — no communication needed. Attention then runs locally: each GPU performs
scaled dot-product attention for its heads only, producing a local context output of shape
`[batch, seq, heads_per_gpu × head_dim]`.

The output projection `W_O` has shape `[d_model, d_model]` — it maps the concatenated
head outputs back to `d_model`. In full (single-GPU) attention this is:

```
concat = [head_0 | head_1 | ... | head_H]   shape [batch, seq, H × head_dim]  =  [batch, seq, d_model]
output = concat @ W_O                        shape [batch, seq, d_model]
```

With TP=4, GPU i already holds only `head_i`'s output — a slice of `concat` spanning rows
`i × (d_model/4)` to `(i+1) × (d_model/4)` of `W_O`. So each GPU multiplies its slice of
`concat` against the corresponding row-shard of `W_O`:

```
GPU0: head_0_out  @ W_O[0      : d/4, :]  →  partial_0   shape [batch, seq, d_model]
GPU1: head_1_out  @ W_O[d/4    : d/2, :]  →  partial_1   shape [batch, seq, d_model]
GPU2: head_2_out  @ W_O[d/2    : 3d/4, :] →  partial_2   shape [batch, seq, d_model]
GPU3: head_3_out  @ W_O[3d/4   : d,   :]  →  partial_3   shape [batch, seq, d_model]
```

Each partial is a full-width `[batch, seq, d_model]` tensor but only a partial sum —
`output = partial_0 + partial_1 + partial_2 + partial_3`. This is exactly the row-parallel
pattern: the input is naturally already split across GPUs (each holds its own head outputs),
so no scatter is needed up front. An all-reduce sums the partials and every GPU gets the
full attention output.

*FFN sub-block:*

The FFN is two linear layers with a non-linearity between them: `FFN(x) = activation(x @ W_1) @ W_2`.
`W_1` is column-parallel: each GPU holds a column shard and computes a different part of the
expanded hidden state independently. The activation function runs locally on each GPU's shard
— no communication. `W_2` is row-parallel: each GPU multiplies its shard of the hidden state
against its row shard of `W_2`, producing a partial sum. An all-reduce sums them, giving every
GPU the full FFN output.

The same TP group is used throughout. Each transformer block contains exactly two all-reduces:
one after the attention output projection, one after the FFN's second linear layer.

### 5.3 The Redundant Computation Problem

After each all-reduce, every GPU in the TP group has the full output tensor of shape
`[batch, seq, d_model]`. The next operation is typically LayerNorm, which normalizes over
the `d_model` dimension for each token. Since every GPU now has the *identical* full tensor,
every GPU computes the exact same LayerNorm — `tp_world_size` identical computations where
only one is needed. Every GPU also holds the full activation in memory: `tp_world_size ×
[batch, seq, d_model]` bytes across the group, with only 1× worth of unique data.

### 5.4 Megatron Sequence Parallel (The Bool Flag) — Eliminating TP Redundancy

Megatron-LM's `sequence_parallel: bool` is a fix specifically for this redundancy within TP.
**It is not the same as the sequence parallelism in Part 6** — the name is unfortunately
shared. Megatron SP is an optimization *inside TP*; Part 6 SP is an independent technique for
long sequences. They are explained separately here to avoid confusion.

The key insight: instead of all-reduce after the row-parallel layer (giving the full result to
every GPU), use **reduce-scatter** (giving each GPU one token shard of the result). Then
LayerNorm, dropout, and the residual connection run on this token shard only — using
1/`tp_world_size` of the activation memory. Before the next column-parallel layer, an
all-gather reconstructs the full tensor on all GPUs.

```
Without Megatron SP:
  row-parallel → all-reduce → [batch, seq, d_model] on all GPUs → LayerNorm (redundant)

With Megatron SP:
  row-parallel → reduce-scatter → [batch, seq/N, d_model] on each GPU → LayerNorm (non-redundant)
              → all-gather → [batch, seq, d_model] on all GPUs → next column-parallel layer
```

The total communication volume is identical: all-reduce = reduce-scatter + all-gather. But
activation memory during LayerNorm is reduced by 1/`tp_world_size`.

**This requires TP ≥ 2.** Without TP there is no row-parallel linear layer and no all-reduce
to replace. Megatron silently sets `sequence_parallel = False` when
`tensor_model_parallel_size ≤ 1`. It is only valid as an optimization on top of TP.

---

## Part 6: Sequence Parallelism — Solving the Long-Sequence Problem

TP and ZeRO address the memory costs of *model parameters* and their associated state —
TP by splitting the weight matrices so each GPU holds only a shard, ZeRO by eliminating
redundant copies across DP replicas. DP does neither: it replicates the full model on every
GPU and is purely about throughput, not model memory. None of the three address the
activation memory cost that grows with sequence length.

The bottleneck is activation memory — the tensors stored during the forward pass so the
backward pass can compute gradients. For long sequences these dominate over model state.

Each transformer layer stores for backward: its input `[batch, S, d_model]`, plus Q, K, V
each of shape `[batch, S, d_model]`. Across L layers:

```
≈ 4 × L × S × d_model × 2 bytes   (layer input + Q + K + V, roughly)

Qwen 4B (36 layers, d_model=4096), S=65536, batch=1:
  4 × 36 × 65536 × 4096 × 2 bytes ≈ 72 GB
```

This grows linearly with S and quickly exceeds GPU memory at 64k–128k tokens.

**What about the O(S²) attention matrix?** FlashAttention eliminates it — it never writes
the `[S, S]` matrix to GPU memory. Instead, FlashAttention's backward recomputes attention
tiles on-the-fly from Q, K, V (which are kept). So the attention matrix is *not* the memory
problem; it is already solved. The problem is Q, K, V and the layer inputs, which all scale
as O(S).

SP splits this activation memory across GPUs. In both Ulysses and Ring Attention, each GPU
holds only `1/P` of the sequence tokens for most of the computation, so per-GPU activation
memory scales as `O(S/P)` instead of `O(S)`. SP also parallelises the attention arithmetic
itself (each GPU computes attention for only its token or head subset), but the primary
motivation at long context is the activation memory reduction.

Sequence parallelism distributes the attention computation across GPUs so each GPU handles
only a fraction. The set of GPUs collaborating on the same sequence is called the **SP group**,
and its size is written as `sp_world_size` (or `P` in the formulas below).

There are two fundamentally different techniques:

### 6.1 All-to-All SP (DeepSpeed Ulysses)

**The key insight:** attention can be parallelized across the *head* dimension. If you have
H attention heads and P GPUs, each GPU can independently compute attention for H/P heads —
but it needs the *full sequence* to do so. The all-to-all rearranges the data so each GPU
gets the full sequence for its head subset, computes attention, then the second all-to-all
undoes the rearrangement.

**Starting point:** the input sequence (S tokens) is split across P GPUs before the model
starts. Each GPU holds a token shard: `S/P` tokens, all H heads.

**Concrete example with S=8 tokens, H=4 heads, P=4 GPUs, head_dim=64.**

Tensor shapes are written as `[seq, heads, head_dim]` (batch omitted for clarity).

After QKV projection (runs locally on each GPU's token shard):
```
         seq  heads  head_dim   content
GPU0: Q [ 2,   4,     64 ]     tokens 0-1, all 4 heads
GPU1: Q [ 2,   4,     64 ]     tokens 2-3, all 4 heads
GPU2: Q [ 2,   4,     64 ]     tokens 4-5, all 4 heads
GPU3: Q [ 2,   4,     64 ]     tokens 6-7, all 4 heads
```
(K and V have the same shape.)

Each GPU has a short slice of the sequence (2 tokens) but all 4 heads. For attention,
we need the opposite: the full sequence but only a subset of heads. The all-to-all
performs this swap.

**How the all-to-all works — GPU0's perspective:**

GPU0 splits its tensor along the head dimension into 4 equal chunks (one per GPU) and
sends each chunk to the corresponding GPU:
```
GPU0 sends to GPU0 (keeps): tokens 0-1, head 0  →  [ 2, 1, 64 ]
GPU0 sends to GPU1:         tokens 0-1, head 1  →  [ 2, 1, 64 ]
GPU0 sends to GPU2:         tokens 0-1, head 2  →  [ 2, 1, 64 ]
GPU0 sends to GPU3:         tokens 0-1, head 3  →  [ 2, 1, 64 ]
```
At the same time, GPU0 receives from every other GPU — but only head 0 from each:
```
GPU0 receives from GPU0 (self): tokens 0-1, head 0  →  [ 2, 1, 64 ]
GPU0 receives from GPU1:        tokens 2-3, head 0  →  [ 2, 1, 64 ]
GPU0 receives from GPU2:        tokens 4-5, head 0  →  [ 2, 1, 64 ]
GPU0 receives from GPU3:        tokens 6-7, head 0  →  [ 2, 1, 64 ]
```
GPU0 concatenates these 4 chunks along the sequence dimension:
`[ 2, 1, 64 ] × 4 → [ 8, 1, 64 ]` — all 8 tokens, head 0 only.

After the all-to-all, the layout across all GPUs is:
```
         seq  heads  head_dim   content
GPU0: Q [ 8,   1,     64 ]     all 8 tokens, head 0 only
GPU1: Q [ 8,   1,     64 ]     all 8 tokens, head 1 only
GPU2: Q [ 8,   1,     64 ]     all 8 tokens, head 2 only
GPU3: Q [ 8,   1,     64 ]     all 8 tokens, head 3 only
```
(Same for K and V.)

The trade: each GPU went from `[S/P, H, head_dim]` to `[S, H/P, head_dim]`. Short sequence +
all heads → full sequence + one head. Now each GPU can run attention independently for its
head over the complete sequence with no further communication.

Each GPU now runs attention locally over its full 8-token sequence for its 1 head.
Recall attention is: `softmax(Q @ K^T / sqrt(head_dim)) @ V`

- `Q` has shape `[seq=8, heads=1, head_dim=64]`. Reshape to `[heads=1, seq=8, head_dim=64]`
  for the matrix multiply.
- `Q @ K^T` multiplies `[1, 8, 64]` by `[1, 64, 8]` → **attention weights** `[1, 8, 8]`
  — one score for every (query token, key token) pair, for this one head.
  (The `[batch, 1, 8, 8]` in full notation is `[batch, heads, seq_q, seq_k]`.)
- After softmax, multiply by `V` of shape `[1, 8, 64]`:
  `[1, 8, 8] @ [1, 8, 64]` → **attention output** `[1, 8, 64]`
  — for each of the 8 query tokens, a weighted sum of value vectors.
  (The `[batch, 8, 1, 64]` in full notation is `[batch, seq, heads, head_dim]`.)

No cross-GPU communication during attention — each GPU computes its head's full attention
independently.

After attention, a second all-to-all within the same SP group inverts the transformation:
scatter the sequence dimension, gather the head dimension. Each GPU returns to its token shard
with all heads. From here, MLP and LayerNorm run locally with no further communication.

**Memory gain:** the actual memory saving from SP is on the Q, K, V tensors and the attention
output — not the `[S, S]` attention weight matrix, which FlashAttention never materializes
regardless of SP. Without SP, each GPU holds Q, K, V each of shape `[S, H, head_dim]` — the
full sequence and all heads. With SP, each GPU holds Q, K, V of shape `[S, H/P, head_dim]`
during attention — the full sequence but only its head subset. The Q/K/V memory is reduced
by 1/P. More importantly, outside of attention (MLP, LayerNorm), each GPU holds only
`[S/P, d_model]` — its token shard — so activation memory across the whole layer scales as
O(S/P) rather than O(S).

**Hard constraint:** H must be divisible by P, because after the all-to-all each GPU must hold
an integer number of heads. If H=10 and P=4, you'd need 10/4=2.5 heads per GPU — impossible.
This is why ms-swift uses a GCD to automatically find the largest valid P (Part 7).

### 6.2 Ring Attention (Context Parallelism)

A completely different approach: keep each GPU on its token shard throughout, including inside
the attention computation. The SP group size P still applies — but instead of an all-to-all
that rearranges tokens and heads, KV tensors are passed peer-to-peer in a ring.

Each GPU has `S/P` query tokens and its own `S/P` key-value tokens. It needs to compute
attention for its queries against *all* S key-value tokens. Since no GPU holds all of them,
they pass KV tensors around the ring while accumulating partial attention results.

**Concrete example with S=8, P=4:**
```
Initial: GPU0 has tokens 0-1, GPU1 has tokens 2-3, GPU2 has tokens 4-5, GPU3 has tokens 6-7.
```

Step 1 (all GPUs simultaneously):
```
GPU0: compute attention(queries[0-1], kv[0-1])   →  pass kv[0-1] to GPU3
      receive kv[6-7] from GPU1
GPU1: compute attention(queries[2-3], kv[2-3])   →  pass kv[2-3] to GPU0
      receive kv[0-1] from GPU2
(etc.)
```
Step 2:
```
GPU0: compute attention(queries[0-1], kv[6-7])   →  pass kv[6-7] to GPU3
      receive kv[4-5] from GPU1
```
Steps 3 and 4 follow the same pattern. After P=4 steps, GPU0 has attended its queries[0-1]
against all 8 tokens' KV — the full attention output for tokens 0-1.

**Important implementation detail — incremental softmax:** standard attention computes
`softmax(QK^T/√d) × V`, which requires seeing all S key tokens at once to normalize correctly.
In ring attention you see one KV block at a time. The solution is the "online softmax"
algorithm: maintain a running maximum and running sum, and update them incrementally using
log-sum-exp arithmetic as each new KV block arrives. This produces numerically identical
results to computing attention over all tokens at once — it is not an approximation.

**No constraint on head count.** KV tensors are passed as whole blocks; the head dimension
is never split.

**Communication volume:** each GPU passes its `[batch, S/P, H, head_dim]` KV block around
the ring P-1 times. Total data moved per GPU ≈ `(P-1)/P × [batch, S, H, head_dim]`.
For comparison, the Ulysses all-to-all moves a similar volume in one bulk step. Ring attention
requires P-1 sequential communication rounds; Ulysses requires 2 (one before attention,
one after). For small P (like 4), Ulysses is typically faster because the bulk all-to-all
can use high-bandwidth interconnects efficiently in one shot.

### 6.3 Comparison

| | All-to-All SP (Ulysses) | Ring Attention (Context Parallelism) |
|---|---|---|
| Named group | SP group, size `sp_world_size` | RP group, size `rp_world_size` |
| Head constraint | H % P == 0 (mandatory) | None |
| Communication steps | 2 (one before attn, one after) | P-1 sequential passes |
| What is sharded during attn | Heads (each GPU: full seq, H/P heads) | Tokens (each GPU: S/P queries, all heads) |
| Best for | P small, H cleanly divisible by P | P large, or H not divisible by P |
| Memory gain (activations) | During attn: Q/K/V are `[S, H/P, d_head]` (full seq, fewer heads). Outside attn: `[S/P, d_model]`. Overall O(S/P). | During attn: Q/K/V are `[S/P, H, d_head]` (fewer tokens, all heads). Outside attn: `[S/P, d_model]`. Overall O(S/P). |

---

## Part 7: How ms-swift Combines Both Techniques (The GCD Logic)

ms-swift's `sequence_parallel_size` config key sets the total size of the SP group — how many
GPUs collaborate on a single sequence. Internally, ms-swift automatically splits this into an
Ulysses component (called `sp_world_size`) and a Ring component (called `rp_world_size`):

```python
sp_world_size = gcd(num_heads, sequence_parallel_size)   # Ulysses sub-group size
rp_world_size = sequence_parallel_size // sp_world_size  # Ring sub-group size
```

`sp_world_size` is the largest number that satisfies the Ulysses divisibility constraint
(`H % sp_world_size == 0`) while not exceeding the total requested parallelism. The leftover
goes to ring, which has no constraint. The two together always multiply back to
`sequence_parallel_size`.

**Examples with 4 GPUs (sequence_parallel_size=4):**

```
Qwen3-4B (num_heads=8):
  sp_world_size = gcd(8, 4) = 4   → pure Ulysses; each GPU holds 8/4=2 heads
  rp_world_size = 4/4 = 1         → no ring component

Hypothetical (num_heads=3):
  sp_world_size = gcd(3, 4) = 1   → Ulysses degree 1 (no communication)
  rp_world_size = 4/1 = 4         → pure ring; 4 GPUs pass KV in a ring

Hypothetical (num_heads=10):
  sp_world_size = gcd(10, 4) = 2  → Ulysses across 2 GPUs
  rp_world_size = 4/2 = 2         → Ring across 2 GPUs (hybrid)
```

### 7.1 The Hybrid Case: A 2×2 Device Mesh

When both `sp_world_size > 1` and `rp_world_size > 1` (e.g., sp=2, rp=2 with 4 GPUs), the
SP group of 4 GPUs is internally arranged as a 2×2 grid. Each GPU has two sub-group indices:
its **sp_rank** (its position within the Ulysses sub-group, 0 to sp_world_size-1) and its
**rp_rank** (its position within the Ring sub-group, 0 to rp_world_size-1).

```
              sp_rank=0    sp_rank=1
rp_rank=0:    GPU 0        GPU 1
rp_rank=1:    GPU 2        GPU 3
```

The **Ulysses (SP) sub-groups** are the rows — GPUs that share the same rp_rank:
`{GPU0, GPU1}` and `{GPU2, GPU3}`. The all-to-all communication runs within each row.

The **Ring (RP) sub-groups** are the columns — GPUs that share the same sp_rank:
`{GPU0, GPU2}` and `{GPU1, GPU3}`. The ring KV passing runs within each column.

Every GPU is a member of both a Ulysses sub-group (one row) and a Ring sub-group (one column).
All 4 GPUs participate in both communications — there is no partition of "these GPUs do Ulysses,
those do ring." Every GPU does both, just with different partners for each.

**Forward pass trace for GPU0 (starting state: tokens 0–S/4, all H heads):**

1. QKV projection runs locally. GPU0 holds `[batch, S/4, H, head_dim]`.

2. **All-to-all within Ulysses sub-group {GPU0, GPU1}** (GPU0 and GPU1 swap head/token
   dimensions):
   - Before: GPU0=`[batch, S/4, H, head_dim]`, GPU1=`[batch, S/4, H, head_dim]`
   - After:  GPU0=`[batch, S/2, H/2, head_dim]`, GPU1=`[batch, S/2, H/2, head_dim]`
   - GPU0 now has tokens 0–S/2 but only heads 0 to H/2-1.

3. **Ring attention within Ring sub-group {GPU0, GPU2}** (2 steps, since rp=2):
   - Step 1: GPU0 attends queries[0–S/2] against its own KV[0–S/2]. Passes KV[0–S/2] to GPU2,
     receives KV[S/2–S] from GPU2.
   - Step 2: GPU0 attends queries[0–S/2] against KV[S/2–S].
   - GPU0 now has the complete attention output for tokens 0–S/2, heads 0 to H/2-1.

4. **Reverse all-to-all within Ulysses sub-group {GPU0, GPU1}**:
   - Returns GPU0 to `[batch, S/4, H, head_dim]` — token shard with all heads.

5. MLP and LayerNorm run locally on the `[batch, S/4, H, head_dim]` token shard.

---

## Part 8: Why All Three Parallelisms Are Sometimes Needed Together

Each parallelism technique solves a different bottleneck:

- **TP** is needed when the model's weight matrices don't fit on a single GPU. For a 70B model,
  the Q projection weight matrix alone is `[8192, 8192]` = 512M parameters = 1 GB in bfloat16.
  A full transformer layer across Q, K, V, O, and two FFN weights is many GB — for very large
  models this doesn't fit in one GPU's 80 GB.

- **SP** (Ulysses/Ring) is needed when a single sequence's attention computation doesn't fit
  on one GPU. For S=128k and 8 heads, the attention matrix per head is `[128k, 128k]` = 32 GB
  in float16 — clearly impossible on one GPU.

- **DP + ZeRO** is needed to efficiently use all remaining GPU capacity. Once you've assigned
  enough GPUs for TP and SP, leftover capacity should be used for data parallelism.

You might need all three: model too big for one GPU (TP), sequence too long for one GPU (SP),
still have remaining GPUs to fill (DP + ZeRO).

### 8.1 A Concrete 3D Example: 16 GPUs, TP=2, SP=4, DP=2

Total GPU count check: TP × SP × DP = 2 × 4 × 2 = 16. ✓

Each GPU gets a 3-tuple address `(dp_rank, tp_rank, sp_rank)`, where each index is that GPU's
position within its respective group:

```
(dp=0, tp=0): GPU0 (sp=0), GPU1 (sp=1), GPU2 (sp=2), GPU3 (sp=3)
(dp=0, tp=1): GPU4 (sp=0), GPU5 (sp=1), GPU6 (sp=2), GPU7 (sp=3)
(dp=1, tp=0): GPU8 (sp=0), GPU9 (sp=1), GPU10(sp=2), GPU11(sp=3)
(dp=1, tp=1): GPU12(sp=0), GPU13(sp=1), GPU14(sp=2), GPU15(sp=3)
```

From this layout, the three groups for any given GPU can be read off directly:

**SP groups** — same `(dp_rank, tp_rank)`, all 4 sp_ranks: GPUs that collaborate on the same
sequence. Size=4. Example: {GPU0, GPU1, GPU2, GPU3} for dp=0, tp=0.

**TP groups** — same `(dp_rank, sp_rank)`, both tp_ranks: GPUs that share weight matrix shards.
Size=2. Example: {GPU0, GPU4} for dp=0, sp=0. The all-reduces in TP run within these pairs.

**DP groups** — same `(tp_rank, sp_rank)`, both dp_ranks: GPUs that are data-parallel replicas
of each other. Size=2. Example: {GPU0, GPU8} for tp=0, sp=0. ZeRO shards across these pairs.

What each GPU holds:
- 1/2 of each weight matrix (TP shard)
- 1/4 of the input sequence tokens (SP shard)
- 1/2 of the LoRA adapter gradients/optimizer states (ZeRO shard within DP group)

### 8.2 Interaction: ZeRO and SP Together

Whether ZeRO and SP operate on the same or different process groups depends on how SP is
registered with DeepSpeed — specifically, whether an mpu is passed (see section 4.5).

**With mpu (Megatron-DeepSpeed):** ZeRO uses the DP process group (size = world_size / sp_size).
SP uses the SP process group. They are separate groups, operate on separate tensors, and do not
interfere. When sp = total_gpus, the DP group has size 1 and ZeRO above stage 0 is useless.

**Without mpu (ms-swift):** DeepSpeed uses the flat world group as its ZeRO group, which
happens to be the same 4 ranks as the SP group. ZeRO shards and reduces across all SP ranks.
This works correctly because ms-swift's custom SP loss backward compensates for DeepSpeed's
automatic divide-by-world-size normalization (see section 4.5 for the full explanation).
`zero3_offload` with `sequence_parallel_size == total_gpus` is a standard, correct
configuration in ms-swift.

### 8.3 Interaction: ZeRO3 and TP

Incompatible. Both ZeRO3 and TP shard weight matrices, but along different dimensions for
different purposes. TP shards across the TP group (columns or rows of the matrix). ZeRO3
shards across the DP group (the full parameter tensor, whatever shape it has). Applying both
would require each GPU to hold a shard-of-a-shard, needing two different sets of communication
partners to reconstruct two different dimensions — a pattern no framework implements correctly.
ms-swift enforces this: `deepspeed_autotp_size` requires ZeRO stage ≤ 2 and raises an
assertion error if violated.

### 8.4 Interaction: Megatron SP (the bool) and TP

Tightly coupled by design — Megatron SP is an internal optimization on top of TP's row-parallel
all-reduce, not a standalone technique. There is no all-reduce to replace when TP=1, so
Megatron silently sets `sequence_parallel = False` when `tensor_model_parallel_size ≤ 1`.

### 8.5 Pipeline Parallelism (PP)

A fourth type of parallelism that splits the model *by layer* across GPUs. GPU0 runs layers
1–8, GPU1 runs layers 9–16, GPU2 runs layers 17–24, GPU3 runs layers 25–32. Each GPU holds
only its layer shard's parameters.

Communication: after GPU0 finishes its layers, it sends the activation tensor to GPU1 via a
direct send/receive (not an all-reduce). During backward, gradients flow backwards in reverse.

The main challenge: GPU1 cannot start until GPU0 finishes its layers. This creates *pipeline
bubbles* — GPU1, GPU2, GPU3 are idle while GPU0 is computing. Microbatching (splitting the
batch into many small microbatches and overlapping their pipeline stages) reduces but does not
eliminate bubble overhead.

PP appears in Megatron via `pipeline_model_parallel_size` but is not used in this repo.

---

## Part 9: LoRA Fine-Tuning and How It Changes the Memory Picture

This repo uses LoRA (Low-Rank Adaptation), a parameter-efficient fine-tuning technique.
Instead of training all of a model's parameters, LoRA adds small trainable "adapter" matrices
alongside each frozen weight matrix. For a frozen weight matrix `W` of shape `[d, d]`, LoRA
inserts `W_delta = A × B` where A has shape `[d, r]` and B has shape `[r, d]`, with rank
`r << d` (here r=16). During the forward pass, the effective weight is `W + A × B`. Only
`A` and `B` are updated; `W` remains frozen.

**Why this dramatically changes the memory picture:**

For Qwen3-4B (4 billion parameters, ~8 GB in bfloat16):

| What | Without LoRA | With LoRA (r=16) |
|---|---|---|
| Base parameters | 8 GB (bfloat16) | 8 GB (bfloat16, frozen, no grads) |
| Gradients | 8 GB (bfloat16) | ~50 MB (only for LoRA adapters) |
| Adam moments (fp32) | 32 GB | ~200 MB (only for LoRA adapters) |
| **Total model state** | **48 GB** | **~8.3 GB** |

ZeRO partitions optimizer states and gradients only for trainable parameters (the optimizer
step filters by `requires_grad`). With LoRA, that means ~50 MB of adapter gradients and
~200 MB of adapter optimizer states — negligible. However, ZeRO-3 *does* shard frozen
parameters at rest and all-gathers them before every layer, so the 8 GB of base weights is
spread across GPUs (2 GB per GPU with 4 GPUs). The all-gather runs for every layer including
frozen ones — communication overhead scales with model depth, not just trainable parameter count.

This is why the eval pass in this repo used only 33.5 GiB per GPU despite a 4B model: the base
weights (8 GB) plus activations for 2 sequences at 64k context length (most of the remaining
memory) fit — there is no 32 GB optimizer state.

The backward pass adds: gradient buffers propagated through frozen layers (required to reach
the LoRA adapters), gradient-checkpointing recomputation, and ZeRO3's reduce-scatter buffers.
That combination pushed over the 40 GB GPU limit.

**Why `deepspeed_autotp_size` doesn't work with LoRA:** TP shards the weight matrix W
(column-parallel or row-parallel), changing its physical shape on each GPU. LoRA's adapters
A and B are defined relative to W's original full shape. When W is sharded across GPUs, the
`W + A×B` addition no longer works straightforwardly — the matrix shapes are inconsistent.
Full-parameter fine-tuning doesn't have this problem because W itself is trained and can be
sharded freely.

---

## Part 10: Putting It Together — Configuration Decisions

The parallelism techniques in Parts 3–9 are controlled by a small set of launch parameters.
The key ones and what they mean:

| Parameter | What it sets |
|---|---|
| `--nproc_per_node N` | Total GPUs per node |
| `--sequence_parallel_size P` | SP group size — how many GPUs collaborate on one sequence |
| `--deepspeed STAGE` | ZeRO stage: `zero0`, `zero1`, `zero2`, `zero3`, `zero3_offload` |
| `--tensor_parallel_size T` | TP group size (not available in all frameworks) |

Everything else is derived:
```
dp_world_size = total_gpus / sequence_parallel_size
```

### 10.1 Choosing SP Size

The SP size determines how the sequence is split and what GPU groups form:

**sp = total_gpus (e.g. sp=4 on 4 GPUs):**
```
SP group: {GPU0, GPU1, GPU2, GPU3}  — all 4 collaborate on one sequence
dp_world_size = 4/4 = 1            — no data parallelism
Per-GPU sequence slice: S/4 tokens
```
Maximises sequence splitting — each GPU sees only S/4 tokens of activation memory.
ZeRO sharding gives no benefit (dp=1); use `zero0`.

**sp = total_gpus / 2 (e.g. sp=2 on 4 GPUs):**
```
SP groups: {GPU0,GPU1} and {GPU2,GPU3}  — two independent SP pairs
DP groups: {GPU0,GPU2} and {GPU1,GPU3}  — dp_world_size = 2
Per-GPU sequence slice: S/2 tokens
```
Each GPU sees S/2 tokens (twice the activation memory vs sp=4), but ZeRO now has
dp=2 peers to shard across. For LoRA the optimizer state is tiny so the activation
cost increase usually outweighs the ZeRO benefit.

**sp = 1 (no sequence parallelism):**
```
DP group: {GPU0, GPU1, GPU2, GPU3}  — dp_world_size = 4
Per-GPU sequence: full S tokens
```
Maximum ZeRO sharding benefit, but each GPU must fit the full sequence in memory.
For long contexts (64k+) this is almost always an OOM.

### 10.2 Choosing ZeRO Stage

**In ms-swift, DeepSpeed always uses the full world group for ZeRO** (no mpu is passed). This
means ZeRO shards across all GPUs regardless of sp_size. All ZeRO stages are valid with any
sp_size. The choice is about memory vs communication overhead:

```
zero0          →  no sharding; DeepSpeed engine active (needed for Ulysses SP)
                  lowest communication overhead

zero2          →  gradients + optimizer states sharded across all world_size GPUs
                  good for full fine-tuning; LoRA optimizer states are tiny so benefit is small

zero3          →  adds parameter sharding on top of zero2
                  every layer triggers all-gather during forward AND backward
                  (including frozen base model weights — significant communication cost)

zero3_offload  →  zero3 + CPU offload of optimizer states
                  frees GPU memory at cost of CPU↔GPU transfer per optimizer step
                  standard choice for long-context LoRA training
```

**Note on LoRA specifically:** LoRA optimizer states are ~200 MB total for a 4B model (only
adapter weights are trained). ZeRO-2's gradient/optimizer state sharding saves ~50 MB per GPU
— negligible. ZeRO-3's parameter sharding saves ~2 GB per GPU on the frozen base weights, but
adds 72 all-gathers per step (forward + backward for 36 layers). `zero3_offload` moves the
small optimizer state to CPU and keeps GPU free for activations — the typical choice.

### 10.3 Worked Examples — 4 GPUs, 64k context, Qwen3-4B (H=8 heads)

*Config A — sp=4, zero3_offload (recommended):*
```
torchrun --nproc_per_node=4 train.py \
  --sequence_parallel_size 4 \
  --deepspeed zero3_offload

SP group: all 4 GPUs (Ulysses, gcd(8,4)=4 → pure Ulysses, each GPU gets 2 heads)
ZeRO group: all 4 GPUs (no mpu → DeepSpeed uses flat world)
Per-GPU tokens outside attention: 64k/4 = 16k
ZeRO-3: frozen base weights sharded across 4 GPUs (~2 GB/GPU saved)
         optimizer states offloaded to CPU
Loss normalization: ms-swift's GatherLoss.backward × world_size compensates for ZeRO's /4
```

*Config B — sp=4, zero0:*
```
torchrun --nproc_per_node=4 train.py \
  --sequence_parallel_size 4 \
  --deepspeed zero0

Same SP setup as Config A
ZeRO: no sharding, no all-gather overhead for frozen weights
Each GPU holds the full 8 GB of base weights at all times
Lower communication cost than Config A; higher peak GPU memory
```

*Config C — sp=2, zero3:*
```
torchrun --nproc_per_node=4 train.py \
  --sequence_parallel_size 2 \
  --deepspeed zero3

SP groups: {GPU0,GPU1}, {GPU2,GPU3} — ms-swift dp_world_size=2 (for data loading)
ZeRO group: all 4 GPUs (DeepSpeed sees flat world)
Per-GPU tokens outside attention: 64k/2 = 32k  ← twice Config A/B
ZeRO-3 shards across 4 GPUs — same sharding as Config A despite sp=2
Net: more activation memory than sp=4, similar ZeRO benefit
```

*Config D — sp=1, zero3:*
```
torchrun --nproc_per_node=4 train.py \
  --sequence_parallel_size 1 \
  --deepspeed zero3

No SP — each GPU holds the full 64k sequence of activations
ZeRO-3: 4× sharding across all 4 GPUs (same as above — DeepSpeed uses world group)
Result: almost certainly OOM at 64k context — activation memory dominates
```

### 10.4 Summary: sp_size and ZeRO in ms-swift

```
ms-swift never passes mpu to DeepSpeed.
DeepSpeed's ZeRO group is always the full world (all GPUs).
ZeRO shards across all GPUs regardless of sp_size.

sp_size controls:  activation memory (S/sp_size tokens per GPU outside attention)
                   which GPUs load which samples (ms-swift's internal dp_world_size)

ZeRO stage controls: whether model state (params/grads/optimizer) is sharded across GPUs
                     and where (GPU vs CPU)

They are orthogonal choices. The correct gradient normalization under SP+ZeRO
is handled by ms-swift's custom loss backward, not by the user.
If sp_size == 1           →  full ZeRO benefit but no activation relief; only viable at short context
```

### 10.5 Combining SP and TP

SP and TP can be used together — but they consume GPUs from the same fixed pool, so
they trade off against each other.

Each GPU must belong to exactly one SP group and one TP group simultaneously. The total
GPUs consumed by the two is multiplicative:

```
GPUs per SP+TP unit = sp_size × tp_size
dp_world_size = total_gpus / (sp_size × tp_size)
```

On 8 GPUs with sp=2 and tp=2:
```
Each "unit" needs 2×2 = 4 GPUs.
8 / 4 = 2 DP replicas.

GPU assignment (one possible layout):
  Unit 0 (DP replica 0): GPU0, GPU1 are SP partners; GPU0, GPU2 are TP partners
  Unit 1 (DP replica 1): GPU4, GPU5 are SP partners; GPU4, GPU6 are TP partners
```

The within-unit communication: Ulysses all-to-all runs between SP partners; TP all-reduce
runs between TP partners. These are different GPU pairs so they do not interfere.

**When does this make sense?**

- **SP** reduces activation memory — useful when the sequence is long.
- **TP** reduces weight matrix memory and compute per GPU — useful when the model is large.
- Using both addresses both bottlenecks simultaneously at the cost of fewer DP replicas
  (lower data throughput).

**Constraint: LoRA and TP are incompatible** (explained in Part 9). If you are doing
LoRA fine-tuning, TP cannot be used. SP alone (without TP) is the correct choice.

**Constraint: ZeRO stage ≤ 2 when using TP.** ZeRO3 shards parameters across the DP group
while TP also shards parameters across the TP group — two different sharding schemes on the
same weights conflict. ZeRO3 is not compatible with TP; use ZeRO0, ZeRO1, or ZeRO2 instead.
