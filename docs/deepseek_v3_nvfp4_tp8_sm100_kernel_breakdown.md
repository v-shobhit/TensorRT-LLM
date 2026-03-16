# DeepSeek-V3.2 NVFP4 on TP8 / SM100 (B200) — Kernel Breakdown

This document provides a high-level submodule breakdown and per-submodule kernel
breakdown for a single inference request through the DeepSeek-V3.2 model running
with NVFP4 quantization, 8-way tensor parallelism, on NVIDIA B200 (SM100).

---

## 1. Model Architecture Overview (DeepSeek-V3 / V3.2)

| Parameter                    | Value                         |
|------------------------------|-------------------------------|
| Hidden size                  | 7168                          |
| Num hidden layers            | 61                            |
| Dense layers (first-k)       | Layers 0–2 (3 dense layers)   |
| MoE layers                   | Layers 3–60 (58 MoE layers, every layer) |
| Attention type               | Multi-head Latent Attention (MLA) |
| Num attention heads          | 128                           |
| q_lora_rank                  | 1536                          |
| kv_lora_rank                 | 512                           |
| qk_nope_head_dim             | 128                           |
| qk_rope_head_dim             | 64                            |
| v_head_dim                   | 128                           |
| Routed experts               | 256                           |
| Experts per token (top-k)    | 8                             |
| Shared experts               | 1 (with intermediate_size = moe_intermediate_size) |
| MoE intermediate size        | 2048                          |
| Dense MLP intermediate size  | 18432                         |
| Vocab size                   | 129280                        |
| Quantization                 | NVFP4 (E2M1 weights + FP8 block scales) |

With **TP8**, each GPU handles:
- **Attention**: 128/8 = 16 heads per rank
- **MoE experts**: 256 experts with weights sharded along intermediate dim by TP8
- **Shared experts**: TP-sharded GatedMLP (tp_size determined by intermediate_size / block_scale_size)

---

## 2. High-Level Submodule Breakdown (Single Forward Pass)

A single inference request flows through these submodules in order:

```
┌──────────────────────────────────────────────────────────────────┐
│  1. EMBEDDING LOOKUP                                            │
│     embed_tokens(input_ids) → hidden_states [BF16]              │
├──────────────────────────────────────────────────────────────────┤
│  2. DECODER LAYERS × 61  (layers 0-60)                          │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  2a. INPUT LAYERNORM  (RMSNorm)                            │  │
│  │      — fused with residual add                             │  │
│  ├────────────────────────────────────────────────────────────┤  │
│  │  2b. SELF-ATTENTION  (MLA)                                 │  │
│  │      ├─ kv_a_proj_with_mqa  (fused Linear: q_a + kv_a + k_pe) │
│  │      ├─ q_a_layernorm + kv_a_layernorm  (parallel RMSNorm)│  │
│  │      ├─ q_b_proj  (Linear, TP-column)                      │  │
│  │      ├─ [Context] kv_b_proj + FMHA (flash attention)       │  │
│  │      ├─ [Gen] BMM1 (q_nope × k_b_proj_trans) ∥ RoPE+KV$   │  │
│  │      ├─ [Gen] MQA FMHA (latent attention)                  │  │
│  │      ├─ [Gen] BMM2 (attn_out × v_b_proj)                   │  │
│  │      └─ o_proj  (Linear, TP-row → AllReduce)               │  │
│  ├────────────────────────────────────────────────────────────┤  │
│  │  2c. POST-ATTENTION LAYERNORM  (RMSNorm)                   │  │
│  │      — fused: AllReduce + residual + RMSNorm               │  │
│  ├────────────────────────────────────────────────────────────┤  │
│  │  2d. FFN (layer-dependent: Dense MLP or MoE)               │  │
│  │                                                            │  │
│  │  [Dense layers 0–2]: GatedMLP                              │  │
│  │      ├─ gate_up_proj  (fused Linear, TP-column)            │  │
│  │      ├─ SwiGLU activation                                  │  │
│  │      └─ down_proj  (Linear, TP-row → AllReduce)            │  │
│  │                                                            │  │
│  │  [MoE layers 3–60]: Deepseekv3MoE                          │  │
│  │      ├─ GATE: router logits  (matmul + DeepSeekV3 routing) │  │
│  │      ├─ ROUTED EXPERTS (in parallel with shared):          │  │
│  │      │   ├─ Token permutation / routing                    │  │
│  │      │   ├─ GEMM1: gate_up (NVFP4 fused MoE GEMM+SwiGLU)  │  │
│  │      │   ├─ GEMM2: down_proj (NVFP4 fused MoE GEMM)       │  │
│  │      │   └─ Token un-permutation / finalize                │  │
│  │      ├─ SHARED EXPERT (on aux stream, parallel):           │  │
│  │      │   ├─ gate_up_proj  (NVFP4 Linear)                   │  │
│  │      │   ├─ SwiGLU activation                              │  │
│  │      │   └─ down_proj  (NVFP4 Linear)                      │  │
│  │      ├─ Combine: routed_output + shared_output             │  │
│  │      └─ AllReduce  (TP reduction)                          │  │
│  ├────────────────────────────────────────────────────────────┤  │
│  │  2e. NEXT-LAYER LAYERNORM  (RMSNorm, fused with residual) │  │
│  │      — fused: AllReduce + residual + RMSNorm (POST_MOE)    │  │
│  └────────────────────────────────────────────────────────────┘  │
├──────────────────────────────────────────────────────────────────┤
│  3. FINAL NORM  (RMSNorm)                                       │
├──────────────────────────────────────────────────────────────────┤
│  4. LM HEAD  (Linear: hidden_size → vocab_size)                 │
│     Output logits for token sampling                            │
└──────────────────────────────────────────────────────────────────┘
```

---

## 3. Per-Submodule Kernel Breakdown

### 3.1 Embedding Lookup

| Kernel | Description |
|--------|-------------|
| `embedding_lookup` | `nn.Embedding` index gather, BF16 output |

---

### 3.2 RMSNorm (appears 3× per decoder layer + 1× final)

Used for: `input_layernorm`, `post_attention_layernorm`, `q_a_layernorm`, `kv_a_layernorm`, final `norm`.

| Kernel | Description |
|--------|-------------|
| `rms_norm_kernel` | Elementwise: x / sqrt(mean(x²) + eps) * weight. BF16 in/out. |
| `fused_residual_rms_norm` | When fused: residual-add + RMSNorm in single kernel pass |
| `allreduce_residual_rmsnorm` | When fused with AllReduce (`AllReduceFusionOp.RESIDUAL_RMS_NORM`): NCCL AllReduce → residual-add → RMSNorm in fused custom kernel |
| `allreduce_residual_rmsnorm_quant_nvfp4` | Variant for dense MLP layers: AllReduce → residual → RMSNorm → NVFP4 quantize (fused, `RESIDUAL_RMS_NORM_QUANT_NVFP4`) |

---

### 3.3 MLA Attention (Multi-head Latent Attention)

#### 3.3.1 Projection GEMMs

| Kernel | Op | Shape (per rank) | Precision |
|--------|----|-------------------|-----------|
| `nvfp4_gemm` / `cutlass_gemm` | `kv_a_proj_with_mqa` | [tokens, 7168] × [7168, 2112] | NVFP4 weights, BF16 act → BF16 out |
| `rms_norm_kernel` × 2 | `q_a_layernorm` + `kv_a_layernorm` | [tokens, 1536] and [tokens, 512] | BF16 (parallel on main + aux stream) |
| `nvfp4_gemm` / `cutlass_gemm` | `q_b_proj` (TP-column) | [tokens, 1536] × [1536, 3072] | NVFP4 weights → BF16 (3072 = 16 heads × 192 head_dim) |
| `nvfp4_gemm` / `cutlass_gemm` | `kv_b_proj` (context only, TP-col) | [tokens, 512] × [512, 4096] | NVFP4 weights → BF16 (4096 = 16 × (128+128)) |

For NVFP4 on SM100, these linear layers use CUTLASS NVFP4 GEMM kernels with:
- E2M1 weight format (2 values packed per byte)
- FP8 block scaling factors (16-element blocks)
- SM100 TMA warp-specialized mainloop (`KernelTmaWarpSpecialized1SmNvf4Sm100` or `2Sm` variant)

#### 3.3.2 Context Phase Attention

| Kernel | Description |
|--------|-------------|
| `fmha_v2_flash_attention` / `flash_fwd_kernel` | Flash Multi-Head Attention (FMHA) for context. Full MHA with K-nope and K-rope concatenated. SM100 uses optimized flash attention kernels. Operates on 16 heads per rank, head_dim=192. |
| `rope_kernel` | YaRN-based rotary position embedding (if not fused into FMHA) |

#### 3.3.3 Generation Phase Attention (Weight Absorption Path)

| Kernel | Description |
|--------|-------------|
| **BMM1** `fp8_block_scaling_bmm` / `trtllm::bmm_out` | Absorb k_b_proj into query: q_nope × k_b_proj_transᵀ → fused_q_nope. Shape: [16 heads, tokens, 128] × [16 heads, 512, 128]ᵀ → [16 heads, tokens, 512]. Uses FP8 block-scaled BMM on SM100 (or BF16 BMM fallback). |
| **RoPE + KV cache** `mla_rope_generation` | Parallel on aux stream: apply rotary embedding to q_pe, prepare latent_cache, compute cumulative sequence lengths. Fused custom kernel. |
| **MQA FMHA** `fmha_v2_mla_generation` / `mlaGeneration` | Latent-space attention: fused_q [tokens, 16 heads, 576] attends to KV cache (latent_cache of dim kv_lora_rank + rope_dim = 576). Single KV head (MQA). Optimized SM100 FMHA kernel. |
| **BMM2** `fp8_block_scaling_bmm` / `trtllm::bmm_out` | Recover from latent space: attn_out × v_b_proj → output. Shape: [16 heads, tokens, 512] × [16 heads, 512, 128]ᵀ → [16 heads, tokens, 128]. FP8 block-scaled BMM on SM100. |

#### 3.3.4 Output Projection

| Kernel | Description |
|--------|-------------|
| `nvfp4_gemm` / `cutlass_gemm` | `o_proj` (TP-row): [tokens, 2048] × [2048, 7168]. NVFP4 weights → BF16. |
| `nccl_allreduce` | AllReduce across 8 TP ranks (unless fused into post-attention norm) |

---

### 3.4 Dense GatedMLP (Layers 0–2)

| Kernel | Description |
|--------|-------------|
| `nvfp4_gemm` | `gate_up_proj` (TP-column, fused gate+up): [tokens, 7168] × [7168, 4608]. NVFP4. (18432*2/8 = 4608 per rank) |
| `swiglu_kernel` | SwiGLU activation: silu(gate) * up. With FP8 output quantization for down_proj input. |
| `nvfp4_gemm` | `down_proj` (TP-row): [tokens, 2304] × [2304, 7168]. NVFP4. |
| `nccl_allreduce` | TP AllReduce (unless fused) |

---

### 3.5 MoE Layer (Layers 3–60) — **Dominates compute**

The MoE layer runs the **routed experts** and **shared expert** in parallel on separate CUDA streams.

#### 3.5.1 Router / Gating

| Kernel | Description |
|--------|-------------|
| `gemm` (gate weight) | Router logits: [tokens, 7168] × [256, 7168]ᵀ → [tokens, 256]. BF16 matmul. |
| `RoutingKernel_DeepSeekV3` | DeepSeekV3-specific routing: sigmoid → routing bias → top-8 selection. Fused routing kernel (TRTLLMGen) or `topk_kernel` + `finalize_moe_route` (CUTLASS backend). |

#### 3.5.2 Routed Experts (256 experts, top-8 per token)

On SM100 with NVFP4, the preferred MoE backends are **TRTLLMGen** or **CutlassFusedMoE**.

**TRTLLMGen Backend (SM100-optimized precompiled kernels):**

| Kernel | Description |
|--------|-------------|
| `RoutingKernel` (DeepSeekV3) | Expert routing with sigmoid-based selection |
| `GemmGatedActKernel_E2m1_E2m1E2m1_..._swiGlu_sm100f` | **Fused GEMM1 + SwiGLU**: token-expert gate_up GEMM with integrated gated activation. NVFP4×NVFP4, precompiled for SM100. Per-rank shape: [tokens_per_expert, 2048/8] from [tokens_per_expert, 7168]. |
| `DevKernel` (GEMM2) | Down projection: [tokens_per_expert, 256] × [256, 7168]. NVFP4. |
| `finalize_moe_routing` | Un-permute tokens and apply expert scaling factors |

**CutlassFusedMoE Backend (CUTLASS MoE GEMM):**

| Kernel | Description |
|--------|-------------|
| `moe_permute` / `topk_softmax` | Token-to-expert routing and permutation |
| `MoeFCGemm` (GEMM1, FP4) | Gate+up: grouped GEMM across active experts. Uses `moe_gemm_kernels_fp4_fp4.cu` or `moe_gemm_kernels_fp8_fp4.cu`. SM100 TMA warp-specialized. |
| `swiglu_kernel` | Gated activation (if not fused into GEMM1) |
| `MoeFCGemm` (GEMM2, FP4) | Down projection: grouped GEMM. NVFP4 weights. |
| `finalize_moe_routing` | Un-permute and scale |

**CuteDslFusedMoE Backend (SM100-native CUTE DSL):**

| Kernel | Description |
|--------|-------------|
| `Sm100BlockScaledContiguousGroupedGemmSwigluFusionRunner` | Fused GEMM1+SwiGLU grouped GEMM |
| `Sm100BlockScaledContiguousGroupedGemmFinalizeFusionRunner` | GEMM2 with finalization |
| `fp8_quantize_1x128` | Dynamic FP8 quantization of activations |

#### 3.5.3 Shared Expert (parallel on aux stream)

| Kernel | Description |
|--------|-------------|
| `nvfp4_gemm` | `gate_up_proj`: [tokens, 7168] × [7168, 512]. NVFP4 (2048*2/8 = 512 per rank). |
| `swiglu_kernel` | SwiGLU + FP8 quantize for down_proj |
| `nvfp4_gemm` | `down_proj`: [tokens, 256] × [256, 7168]. NVFP4. |

#### 3.5.4 Combine + Communication

| Kernel | Description |
|--------|-------------|
| `moe_reduce_add_shared_output` (compiled) | `torch.sum(routed, dim=1) + shared_output`. Compiled elementwise kernel. |
| `nccl_allreduce` | TP AllReduce across 8 ranks. Or fused: |
| `MoEAllReduce` (fused) | When POST_MOE_FUSION enabled with TRTLLMGen: fused un-permute + scale + shared-add + AllReduce + residual + RMSNorm. Single custom kernel avoiding multiple global memory passes. |

---

### 3.6 Final Norm + LM Head

| Kernel | Description |
|--------|-------------|
| `rms_norm_kernel` | Final RMSNorm on hidden_states |
| `gemm` (lm_head) | [tokens, 7168] × [129280, 7168]ᵀ → [tokens, 129280]. BF16 or NVFP4. |

---

## 4. Communication Kernels Summary (TP8)

| Kernel | Where | Frequency per layer |
|--------|-------|---------------------|
| `ncclAllReduce` | After `o_proj` (attention) | 1× |
| `ncclAllReduce` | After MoE/MLP combine | 1× |
| Fused AllReduce+Residual+RMSNorm | Pre-MoE / Post-MoE fusion | Replaces separate kernels when enabled |

Total per decoder layer: **2 AllReduce calls** (may be fused into adjacent operations).

---

## 5. Kernel Fusion Patterns on SM100

TensorRT-LLM applies aggressive fusion for the DeepSeek MoE path on SM100:

### PRE_MOE_FUSION (enabled when TP > 1)
```
AllReduce → Residual Add → RMSNorm → (optional NVFP4 quantize)
```
Replaces 3–4 separate kernels with a single fused kernel.

### POST_MOE_FUSION (enabled when TP > 1, single-node, TRTLLMGen)
```
MoE un-permute → Expert scale → Shared expert add → AllReduce → Residual → RMSNorm
```
The `MoEAllReduce` custom kernel handles the entire post-MoE pipeline.

### MLA Parallelism
- BMM1 (q × k_b_proj_trans) runs in **parallel** with RoPE + KV cache setup on aux stream
- q_a_layernorm and kv_a_layernorm run in **parallel** on main + aux stream
- Routed experts and shared expert run in **parallel** on main + aux stream

---

## 6. Summary: Kernel Count per Decoder Layer

### MoE Layer (layers 3–60)

| Category | Kernels | Notes |
|----------|---------|-------|
| RMSNorm | 3–4 | input_ln, q_a_ln, kv_a_ln, post_attn_ln (some fused) |
| Attention projections | 3–4 GEMMs | kv_a_proj, q_b_proj, (kv_b_proj context only), o_proj |
| Attention core | 1–2 FMHA | Context FMHA and/or Generation MLA FMHA |
| Attention BMMs (gen) | 2 BMMs | k_b absorption + v_b recovery |
| RoPE | 1 | If not fused into FMHA |
| MoE routing | 1–2 | Gate matmul + routing kernel |
| MoE GEMM1 (routed) | 1 | Fused grouped GEMM + SwiGLU |
| MoE GEMM2 (routed) | 1 | Grouped down-projection GEMM |
| MoE finalize | 1 | Un-permute + scaling |
| Shared expert | 3 | gate_up GEMM + SwiGLU + down GEMM |
| Combine + reduce | 1–2 | Sum + AllReduce (potentially fused) |
| Communication | 2 | AllReduce × 2 (attn + MoE) |
| **Total** | **~20–25 kernel launches** | Per MoE decoder layer |

### Dense Layer (layers 0–2)

| Category | Kernels |
|----------|---------|
| Same attention path | ~12–15 |
| Dense GatedMLP | 3 (gate_up + SwiGLU + down) |
| Communication | 2 AllReduce |
| **Total** | **~17–20 kernel launches** |

---

## 7. References

- Model: `tensorrt_llm/_torch/models/modeling_deepseekv3.py`
- MLA: `tensorrt_llm/_torch/modules/attention.py` (class `MLA`)
- MoE backends: `tensorrt_llm/_torch/modules/fused_moe/`
- CUTLASS MoE kernels: `cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/`
- TRTLLMGen MoE: `cpp/tensorrt_llm/kernels/trtllmGenKernels/blockScaleMoe/`
- NVFP4 GEMM SM100: `cpp/tensorrt_llm/kernels/cutlass_kernels/fp4_gemm/nvfp4_nvfp4_gemm_template_sm100.h`
- Fusion ops: `tensorrt_llm/_torch/distributed/` (AllReduce, MoEAllReduce)
