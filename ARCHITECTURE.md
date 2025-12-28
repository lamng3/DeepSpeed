# DeepSpeed Architecture Overview

## Introduction

DeepSpeed is a deep learning optimization library that enables efficient training and inference of large-scale models. This document provides a high-level overview of the codebase architecture.

## Entry Points

The main entry points are in `deepspeed/__init__.py`:

- **`deepspeed.initialize()`**: Initializes the training engine with a PyTorch model
- **`deepspeed.init_inference()`**: Initializes the inference engine for model serving

## Core Components

### 1. Runtime Engine (`deepspeed/runtime/`)

The heart of DeepSpeed training:

- **`engine.py`**: `DeepSpeedEngine` - Main training engine that wraps PyTorch models
- **`hybrid_engine.py`**: `DeepSpeedHybridEngine` - Combines training and inference
- **`pipe/engine.py`**: `PipelineEngine` - Pipeline parallelism support

### 2. Memory Optimization - ZeRO (`deepspeed/runtime/zero/`)

ZeRO (Zero Redundancy Optimizer) partitions optimizer states, gradients, and parameters across GPUs:

- **ZeRO Stage 1**: Optimizer state partitioning
- **ZeRO Stage 2**: Gradient partitioning
- **ZeRO Stage 3**: Parameter partitioning
- **ZeRO-Offload**: Offloads optimizer states to CPU
- **ZeRO-Infinity**: Offloads to CPU and NVMe
- **ZenFlow**: Asynchronous offloading engine

### 3. Parallelism Strategies

- **Data Parallelism**: Built into the engine via distributed groups
- **Pipeline Parallelism**: `deepspeed/pipe/` - Splits model layers across stages
- **Tensor Parallelism**: `deepspeed/runtime/tensor_parallel/` - Partitions tensors
- **Sequence Parallelism**: `deepspeed/sequence/` - Partitions sequence dimension (Ulysses)

### 4. Inference Engine (`deepspeed/inference/`)

Optimized inference for transformer models:

- **`engine.py`**: `InferenceEngine` - Main inference runtime
- **`v2/`**: DeepSpeed-FastGen - Next-generation inference engine
- Kernel injection and quantization support

### 5. Custom Operations (`deepspeed/ops/` & `csrc/`)

High-performance CUDA/C++ kernels:

- **Optimizers**: Fused Adam, LAMB, Lion, Muon
- **Transformers**: Optimized attention and MLP kernels
- **Quantization**: FP6, INT8, INT4 quantization kernels
- **Communication**: Optimized collectives (allreduce, allgather)

### 6. Communication Layer (`deepspeed/comm/`)

Abstraction over distributed communication:

- **`comm.py`**: Communication primitives (allreduce, allgather, etc.)
- **`backend.py`**: Backend selection (NCCL, MPI, etc.)
- **`torch.py`**: PyTorch distributed integration

### 7. Hardware Abstraction (`deepspeed/accelerator/`)

Unified interface for different hardware:

- **`abstract_accelerator.py`**: Base accelerator interface
- **`cuda_accelerator.py`**: NVIDIA CUDA support
- **`cpu_accelerator.py`**: CPU support
- **`hpu_accelerator.py`**: Intel Gaudi support
- **`xpu_accelerator.py`**: Intel GPU support
- **`mps_accelerator.py`**: Apple Metal support
- **`npu_accelerator.py`**: Huawei Ascend support

### 8. Model Transformation (`deepspeed/module_inject/`)

Automatically replaces PyTorch modules with optimized versions:

- **`inject.py`**: Module replacement logic
- **`auto_tp.py`**: Automatic tensor parallelism
- **`containers/`**: Optimized layer implementations
- **`replace_policy.py`**: Policies for model-specific replacements

### 9. Checkpointing (`deepspeed/checkpoint/`)

Efficient model and optimizer state saving/loading:

- **`deepspeed_checkpoint.py`**: DeepSpeed checkpoint format
- **`universal_checkpoint.py`**: Universal checkpoint format
- **`zero_checkpoint.py`**: ZeRO-specific checkpoint handling

### 10. Compilation (`deepspeed/compile/`)

Graph compilation for optimization:

- **`fx.py`**: PyTorch FX graph compilation
- **`inductor.py`**: TorchInductor backend
- **`passes/`**: Compilation passes

### 11. Compression (`deepspeed/compression/`)

Model compression techniques:

- Weight quantization
- Pruning
- Knowledge distillation

### 12. Mixture of Experts (`deepspeed/moe/`)

MoE layer implementation for sparse models:

- Expert routing
- Load balancing
- Communication optimization

### 13. I/O Operations (`deepspeed/io/` & `deepspeed/nvme/`)

Efficient data loading and checkpoint I/O:

- **`io/`**: File I/O abstractions
- **`nvme/`**: NVMe-optimized async I/O (DeepNVMe)

### 14. Launcher (`deepspeed/launcher/`)

Distributed training launcher:

- **`launch.py`**: Multi-node job launching
- **`runner.py`**: Process management

## Data Flow

### Training Flow

```
User Code
  ↓
deepspeed.initialize(model, config)
  ↓
DeepSpeedEngine
  ↓
[ZeRO Partitioning] → [Communication] → [Optimizer Step]
  ↓
Model Update
```

### Inference Flow

```
User Code
  ↓
deepspeed.init_inference(model, config)
  ↓
InferenceEngine
  ↓
[Kernel Injection] → [Tensor Parallelism] → [Quantization]
  ↓
Optimized Inference
```

## Key Design Principles

1. **Minimal API Changes**: Works with existing PyTorch code
2. **Composability**: Features can be combined (ZeRO + Pipeline + Tensor Parallel)
3. **Hardware Agnostic**: Abstracted accelerator interface
4. **Automatic Optimization**: Module injection and kernel replacement
5. **Memory Efficiency**: ZeRO and offloading reduce memory footprint
6. **Communication Efficiency**: Optimized collectives and quantization

## Configuration

DeepSpeed uses JSON configuration files (`deepspeed_config`) that specify:
- ZeRO stage and offloading
- Optimizer and scheduler settings
- Mixed precision (FP16/BF16)
- Pipeline parallelism
- Communication settings
- Checkpointing options

## Integration Points

- **PyTorch**: Core dependency, wraps `nn.Module` and `Optimizer`
- **Transformers**: HuggingFace integration via `module_inject`
- **Megatron-LM**: 3D parallelism support
- **Lightning/Accelerate**: Framework integrations

## File Structure Summary

```
deepspeed/
├── __init__.py              # Main API entry points
├── runtime/                  # Training engine and ZeRO
├── inference/                # Inference engine
├── ops/                      # Python wrappers for custom ops
├── csrc/                     # C++/CUDA source code
├── module_inject/            # Model transformation
├── comm/                     # Communication layer
├── accelerator/              # Hardware abstraction
├── checkpoint/               # Checkpointing
├── compile/                  # Graph compilation
├── compression/              # Model compression
├── moe/                      # Mixture of Experts
├── pipe/                     # Pipeline parallelism
├── sequence/                 # Sequence parallelism
├── io/                       # I/O operations
└── launcher/                 # Distributed launcher
```

## Further Reading

- [DeepSpeed Documentation](https://www.deepspeed.ai/)
- [API Documentation](https://deepspeed.readthedocs.io/)
- [Getting Started Guide](https://www.deepspeed.ai/getting-started/)

