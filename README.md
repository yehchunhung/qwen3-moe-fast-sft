# Qwen3-MoE Fast SFT 🚀

A high-performance Supervised Fine-Tuning (SFT) specifically optimized for Qwen3-MoE models. This project implements cutting-edge techniques to maximize training speed and reduce memory usage for efficient Qwen3-MoE fine-tuning.

## 🎯 Key Features

- **Ultra-fast Qwen3-MoE SFT** with optimized scatter operations
- **Multiple precision formats** (FP8, BF16) for memory efficiency
- **Advanced distributed training** with DeepSpeed ZeRO and FSDP
- **Kernel optimization** with Liger Kernel and Flash Attention
- **Comprehensive efficiency monitoring** with real-time metrics
- **MoE specific optimization** for maximum performance

## 🛠️ Efficiency Technologies

### Distributed Training

**ZeRO-3**

ZeRO (Zero Redundancy Optimizer) is a memory optimization technology that enables training larger models by distributing model states across GPUs. Following the [DeepSpeed official documentation](https://www.deepspeed.ai/docs/config-json) and [HuggingFace tutorial](https://huggingface.co/docs/accelerate/usage_guides/deepspeed#deepspeed-config-file), we can establish a functional configuration to launch distributed training.

However, this basic setup often falls short when training Mixture of Experts (MoE) models, which contain hundreds of experts that a router can dynamically select from during processing.

To significantly enhance training efficiency, we recommend adding  `stage3_module_granularity_threshold` to your configuration. This setting instructs ZeRO-3 to treat a MoE layer as a unit during parameter prefetching. Our experiments demonstrate that this optimization alone can deliver up to **4x improvement in training efficiency** for MoE models.

> [!Tip]
> The exact value of `stage3_module_granularity_threshold` can be determined automatically. So it's okay to set a random value at first, and then ZeRO-3 will tell you the recommended value for better efficiency.

<details>
<summary>Final ZeRO-3 config</summary>

```json
{
    "bf16": {
        "enabled": true
    },
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "none",
            "pin_memory": false
	},
        "offload_param": {
            "device": "none",
            "pin_memory": false
	},
        "overlap_comm": false,
        "contiguous_gradients": true,
        "sub_group_size": 1e9,
        "reduce_bucket_size": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "allgather_bucket_size": 1e8,
        "stage3_param_persistence_threshold": "auto",
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
        "stage3_gather_16bit_weights_on_model_save": true,
        "stage3_module_granularity_threshold": 1175567
    },
    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
    "steps_per_print": 200,
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "wall_clock_breakdown": false
}
```
</details>
<br />

**FSDP**

Fully Sharded Data Parallel (FSDP) was inspired by ZeRO but implemented natively in PyTorch. Like ZeRO, FSDP shards optimizer states, gradients, and model parameters across multiple GPUs, significantly reducing the memory required per device.

In our implementation, we primarily use the `accelerate` library to manage FSDP hyperparameters. This approach simplifies configuration while maintaining performance benefits. For those interested in deeper customization, we recommend exploring the [`accelerate` documentation](https://huggingface.co/docs/accelerate/usage_guides/fsdp).

> [!NOTE]
> The PyTorch team has recently introduced FSDP2, which offers [additional optimizations and capabilities](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html). However, integration with HuggingFace is still in progress. As a result, some advanced features described later may not work seamlessly with FSDP2 at present.

<details>
<summary>Final FSDP config</summary>

```json
{
    "backward_prefetch": "backward_pre",
    "forward_prefetch": "true",
    "activation_checkpointing": true
}
```
</details>
<br />

### Kernel Optimization

**Liger Kernel**

Liger Kernel is a collection of optimized Triton kernels that enhances LLM training efficiency. By replacing standard PyTorch components like `RMSNorm`, `RoPE`, and `CrossEntropy` with specialized implementations, it delivers approximately 20% higher multi-GPU training throughput while reducing memory usage by 60%.

In our implementation, we load models using the specialized wrapper class `AutoLigerKernelforCausalLM` and selectively apply optimizations. In our code, we specifically turn off `swiglu` while maintaining most of optimizations.

**ScatterMoE**

As previously mentioned, HuggingFace's standard implementation of sparse Mixture of Experts (MoE) blocks uses sequential processing, where the model loops through each expert one by one (as seen in the [Qwen3 MoE implementation](https://github.com/huggingface/transformers/blob/8c59cdb3f81f9f4b8e7b8d05b92d40c2e9413788/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py#L251)). This approach becomes highly inefficient when dealing with models containing numerous fine-grained experts, such as Qwen3-30B-A3B with its 128 experts.

To address this performance bottleneck, we utilize [ScatterMoE](https://github.com/shawntan/scattermoe/tree/main), which fundamentally reimagines sparse MoE computation through:

1. Kernel fusion techniques
2. Direct scattered-data processing
3. Parallelized expert computation

Our implementation monkey-patches the original `Qwen3MoeSparseMoeBlock` to process experts in parallel. The performance result demonstrate a significant **60%** improvement in training throughput compared to the HuggingFace native implementation.

On the whole, this optimization is particularly valuable for large-scale MoE models where the router must efficiently distribute computation across many potential experts.

### Mixed Precision Training

**FP8**

FP8 training leverages hardware capabilities in recent NVIDIA GPUs (after Hopper series) to significantly reduce memory usage and increase throughput for large model training. By converting selected model layers from 16-bit to 8-bit, we can accelerate computation while maintaining training quality.

For numerical stability, we utilize [`torchao`](https://github.com/pytorch/ao/tree/main), a PyTorch-driven hackable FP8 backend that allows for precision-selective quantization. Our implementation strategically preserves the following critical components at its original precision (i.e. `torch.bfloat16`) while converting the others to FP8:

- The first and last decoder layers
- All router layers within each decoder

This selective approach is crucial for maintaining training stability. Our experiments demonstrate that preserving router precision in particular helps achieve lower training loss while still benefiting from the throughput advantages of FP8 elsewhere in the model.

For implementation details of our precision-aware monkey-patching approach, you can examine `monkey_patch/fp8.py`.

> [!TIP]
> It's better to activate `torch.compile` when using FP8, as `torchao` relies on it for generating fused casting kernels.
>
> Reference: https://github.com/pytorch/ao/issues/685


## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- CUDA 12.8+ (for GPU support)
- NVIDIA Hopper or newer GPU recommended for FP8

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yehchunhung/qwen3-moe-fast-sft.git
   ```

2. Install all required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Experiment Setup

1. Dataset: [Yukang/LongAlpaca-12k](https://huggingface.co/datasets/Yukang/LongAlpaca-12k)
2. Machine: Nvidia B200 (CUDA 12.8 support)

### Configuration

  Configuration files for ZeRO and FSDP are located in the `config/` directory. Edit these files to adjust distributed backend or hyperparameters as needed.

### Training

1. **Single node:** run
   ```bash
   bash qwen3_sft.sh
   ```
2. **Multi node:** run
   ```bash
   bash qwen3_sft_multinode.sh
   ```
   on each node. Edit the scripts if you need to change dataset paths or config locations.

## 📊 Results

The following tables summarize the throughput and memory usage for various training strategies, as observed in our experiments. **Lower throughput (sec/step) is better.**

**Hardware:** All results below are from Nvidia B200 GPUs (CUDA 12.8).

**ZeRO-3**

| Training Strategy | Micro-batch Size | Sequence Length | Throughput (sec/step) | Peak Memory (GB) |
| ----------------------------------------- | ---------------- | ----------------------- | -------------------- | ---------------- |
| ZeRO3 (w/o prefetching) | 2 | 4096 | ~318s | 150 |
| ZeRO3 | 2 | 4096 | ~84s | 170 |
| ZeRO3 + Liger Kernel | 4 | 8192 | ~96.7s | 157 |
| ZeRO3 + Liger Kernel + FP8 | 4 | 8192 | ~72s | 156 |
| ZeRO3 + Liger Kernel + ScatterMoE | 4 | 8192 | ~57s | 162 |
| ZeRO3 + Liger Kernel + ScatterMoE + FP8 | 4 | 8192 | ~57s | 162 |
| ZeRO3 + Liger Kernel + ScatterMoE + FP8 + Flash Attention 2* | 4 | 8192 | ~33.7s | 161 |

**FSDP**

| Training Strategy | Micro-batch Size | Sequence Length | Throughput (sec/step) | Peak Memory (GB) |
| ----------------------------------------- | ---------------- | ----------------------- | -------------------- | ---------------- |
| FSDP | 4 | 8192 | ~89.8s | 178 |
| FSDP + Liger Kernel | 4 | 8192 | ~76s | 177 |
| FSDP + Liger Kernel + FP8 | 4 | 8192 | ~54.2s | 177 |
| FSDP + Liger Kernel + ScatterMoE | 4 | 8192 | ~63s | 175 |
| FSDP + Liger Kernel + ScatterMoE + FP8 | 4 | 8192 | ~45.5s | 177 |
| FSDP + Liger Kernel + ScatterMoE + FP8 + Flash Attention 2* | 4 | 8192 | ~22s | 177 |

> [!NOTE]
> *Flash Attention 2 is run by old instructions for Nvidia Ampere architecture. But it's way faster than `sdpa` in our experiments. We speculate that some components in `sdpa` are not fully supported in B200.
> 
> Reference: https://github.com/Dao-AILab/flash-attention/issues/1464#issuecomment-3037130228

### Recommended Configuration

Our comprehensive benchmarks demonstrate that integrating all optimization features (Liger Kernel, ScatterMoE, and FP8) significantly accelerates training regardless of which distributed backend you choose.

Compared to baseline implementations, our optimized configurations deliver remarkable performance improvements:

- ZeRO-3: **2.5×** throughput improvement. (Nearly 10× improvement compared to ZeRO-3 without prefetching.)
- FSDP: **4×** throughput improvement.

These results highlight the cumulative benefits of our optimization strategies when properly configured for large-scale MoE model training. We recommend adopting these techniques for any production-scale training of mixture-of-experts architectures.