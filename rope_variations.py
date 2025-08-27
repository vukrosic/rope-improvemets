import torch
import triton
import triton.language as tl
import math
import time
import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt

# Original Triton RoPE kernel from rope.py
@triton.jit
def rope_forward_kernel_v1(
    x_ptr, out_ptr,
    cos_ptr, sin_ptr,
    batch_size, n_heads, seq_len, head_dim,
    stride_x_batch, stride_x_head, stride_x_seq, stride_x_dim,
    stride_out_batch, stride_out_head, stride_out_seq, stride_out_dim,
    stride_cos_seq, stride_cos_dim,
    stride_sin_seq, stride_sin_dim,
    BLOCK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    pid_head = tl.program_id(1) 
    pid_seq = tl.program_id(2)
    
    x_base = x_ptr + pid_batch * stride_x_batch + pid_head * stride_x_head + pid_seq * stride_x_seq
    out_base = out_ptr + pid_batch * stride_out_batch + pid_head * stride_out_head + pid_seq * stride_out_seq
    cos_base = cos_ptr + pid_seq * stride_cos_seq
    sin_base = sin_ptr + pid_seq * stride_sin_seq
    
    for i in range(0, HEAD_DIM // 2):
        x1 = tl.load(x_base + i * 2)
        x2 = tl.load(x_base + i * 2 + 1)
        
        cos_val = tl.load(cos_base + i)
        sin_val = tl.load(sin_base + i)
        
        y1 = x1 * cos_val + x2 * sin_val
        y2 = -x1 * sin_val + x2 * cos_val
        
        tl.store(out_base + i * 2, y1)
        tl.store(out_base + i * 2 + 1, y2)

# V2 Kernel: 2D grid, 1D blocking
@triton.jit
def rope_forward_kernel_v2(
    x_ptr, out_ptr,
    cos_ptr, sin_ptr,
    stride_x_batch, stride_x_head, stride_x_seq, stride_x_dim,
    stride_out_batch, stride_out_head, stride_out_seq, stride_out_dim,
    stride_cos_seq, stride_cos_dim,
    stride_sin_seq, stride_sin_dim,
    n_heads: tl.constexpr, seq_len: tl.constexpr, head_dim: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr
):
    pid_bh = tl.program_id(0)
    pid_s = tl.program_id(1)

    pid_batch = pid_bh // n_heads
    pid_head = pid_bh % n_heads

    x_base = x_ptr + pid_batch * stride_x_batch + pid_head * stride_x_head + pid_s * stride_x_seq
    out_base = out_ptr + pid_batch * stride_out_batch + pid_head * stride_out_head + pid_s * stride_out_seq
    cos_base = cos_ptr + pid_s * stride_cos_seq
    sin_base = sin_ptr + pid_s * stride_sin_seq

    d_offsets = tl.arange(0, BLOCK_SIZE_D)
    half_d = head_dim // 2

    x1_ptr = x_base + d_offsets
    x2_ptr = x_base + half_d + d_offsets

    cos_ptr_ = cos_base + d_offsets
    sin_ptr_ = sin_base + d_offsets

    mask = d_offsets < half_d

    x1 = tl.load(x1_ptr, mask=mask, other=0.0)
    x2 = tl.load(x2_ptr, mask=mask, other=0.0)
    cos = tl.load(cos_ptr_, mask=mask, other=0.0)
    sin = tl.load(sin_ptr_, mask=mask, other=0.0)

    y1 = x1 * cos + x2 * sin
    y2 = -x1 * sin + x2 * cos

    tl.store(out_base + d_offsets, y1, mask=mask)
    tl.store(out_base + half_d + d_offsets, y2, mask=mask)


class RoPE(torch.nn.Module):
    def __init__(self, dim: int, max_seq_len: int, base: float = 10000.0, correct_impl=False):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        self.correct_impl = correct_impl

        if self.correct_impl:
            inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
            t = torch.arange(max_seq_len, dtype=torch.float32)
            freqs = torch.einsum('i,j->ij', t, inv_freq)
            self.register_buffer('cos', freqs.cos(), persistent=False)
            self.register_buffer('sin', freqs.sin(), persistent=False)
        else: # Buggy implementation from llm.py
            angular_freq = (1 / base) ** torch.linspace(0, 1, steps=dim//4, dtype=torch.float32)
            angular_freq = torch.cat([angular_freq, angular_freq.new_zeros(dim//4)])
            t = torch.arange(max_seq_len, dtype=torch.float32)
            theta = torch.einsum("i,j -> ij", t, angular_freq)
            self.register_buffer('cos', theta.cos(), persistent=False)
            self.register_buffer('sin', theta.sin(), persistent=False)

    def forward_pytorch(self, x_BTHD: torch.Tensor) -> torch.Tensor:
        cos = self.cos[None, :x_BTHD.size(-3), None, :]
        sin = self.sin[None, :x_BTHD.size(-3), None, :]
        
        x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        if self.correct_impl:
            y1 = x1 * cos - x2 * sin
            y2 = x1 * sin + x2 * cos
        else: # Buggy implementation from llm.py
            y1 = x1 * cos + x2 * sin
            y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), -1).type_as(x_BTHD)

    def forward_triton_v1(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, n_heads, seq_len, head_dim = x.shape
        output = torch.empty_like(x)
        cos_seq = self.cos[:seq_len].contiguous()
        sin_seq = self.sin[:seq_len].contiguous()
        
        grid = (batch_size, n_heads, seq_len)
        
        rope_forward_kernel_v1[grid](
            x, output,
            cos_seq, sin_seq,
            batch_size, n_heads, seq_len, head_dim,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            output.stride(0), output.stride(1), output.stride(2), output.stride(3),
            cos_seq.stride(0), cos_seq.stride(1),
            sin_seq.stride(0), sin_seq.stride(1),
            BLOCK_SIZE=128,
            HEAD_DIM=head_dim,
        )
        return output

    def forward_triton_v2(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, n_heads, seq_len, head_dim = x.shape
        output = torch.empty_like(x)
        cos_seq = self.cos[:seq_len].contiguous()
        sin_seq = self.sin[:seq_len].contiguous()

        grid = lambda META: (batch_size * n_heads, seq_len)
        
        rope_forward_kernel_v2[grid](
            x, output,
            cos_seq, sin_seq,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            output.stride(0), output.stride(1), output.stride(2), output.stride(3),
            cos_seq.stride(0), cos_seq.stride(1),
            sin_seq.stride(0), sin_seq.stride(1),
            n_heads=n_heads, seq_len=seq_len, head_dim=head_dim,
            BLOCK_SIZE_D=triton.next_power_of_2(head_dim // 2)
        )
        return output

def benchmark_rope_implementations():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu':
        print("Warning: CUDA not available")
        return
    
    configs = [
        (1, 8, 2048, 64),
        (4, 12, 512, 64),
        (8, 16, 1024, 64),
        (1, 8, 2048, 128),
        (2, 16, 512, 128),
        (4, 16, 1024, 128)
    ]
    
    results = []
    
    print("🚀 Benchmarking RoPE Implementations")
    
    for batch_size, n_heads, seq_len, head_dim in configs:
        print(f"\nConfiguration: B={batch_size}, H={n_heads}, L={seq_len}, D={head_dim}")
        
        x = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device, dtype=torch.float16)
        
        # Use correct implementation for fair comparison
        rope_module = RoPE(head_dim, max(seq_len, 4096), correct_impl=True).to(device)
        
        implementations = {
            "Pytorch": rope_module.forward_pytorch,
            "Triton V1": rope_module.forward_triton_v1,
            "Triton V2": rope_module.forward_triton_v2,
        }

        config_results = {'config': f"{batch_size}x{n_heads}x{seq_len}x{head_dim}"}

        for name, func in implementations.items():
            # Warmup
            for _ in range(10):
                _ = func(x)
            torch.cuda.synchronize()
            
            # Benchmark
            n_runs = 100
            start_time = time.time()
            for _ in range(n_runs):
                output = func(x)
            torch.cuda.synchronize()
            exec_time = (time.time() - start_time) / n_runs
            
            config_results[name] = exec_time * 1000 # ms
            print(f"  {name}: {exec_time*1000:.3f} ms")

        # Correctness check against PyTorch
        pytorch_out = implementations["Pytorch"](x)
        for name, func in implementations.items():
            if name != "Pytorch":
                triton_out = func(x)
                max_diff = torch.max(torch.abs(pytorch_out - triton_out)).item()
                config_results[f"max_diff_{name}"] = max_diff

        results.append(config_results)

    return results

def plot_benchmark_results(results):
    if not results:
        return

    configs = [r['config'] for r in results]
    impl_names = [name for name in results[0] if name not in ['config'] and not name.startswith('max_diff')]
    times = {name: [r[name] for r in results] for name in impl_names}

    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    
    x_pos = np.arange(len(configs))
    width = 0.2
    
    for i, name in enumerate(impl_names):
        ax.bar(x_pos + i*width, times[name], width, label=name, alpha=0.8)
    
    ax.set_xlabel('Configuration (BxHxLxD)')
    ax.set_ylabel('Time (ms)')
    ax.set_title('RoPE Implementation Runtime Comparison')
    ax.set_xticks(x_pos + width * (len(impl_names) -1) / 2)
    ax.set_xticklabels(configs, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("rope_benchmark.png")
    print("\n📊 Benchmark plot saved to rope_benchmark.png")

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("❌ CUDA not available.")
        exit(1)
        
    print(f"🔍 Device: {torch.cuda.get_device_name()}")
    
    benchmark_results = benchmark_rope_implementations()
    
    if benchmark_results:
        # Print summary table
        header = ["Config"] + [name for name in benchmark_results[0] if name != 'config' and not name.startswith('max_diff')]
        print("\n" + "| ".join(header))
        print("|".join(["---"] * len(header)))
        for r in benchmark_results:
            row = [r['config']] + [f"{r[name]:.3f}" for name in header[1:]]
            print("| ".join(row))

        plot_benchmark_results(benchmark_results)
