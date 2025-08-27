import torch
import triton
import triton.language as tl
import math
import time
import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt

# Triton RoPE kernel
@triton.jit
def rope_forward_kernel(
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
    # Get program IDs
    pid_batch = tl.program_id(0)
    pid_head = tl.program_id(1) 
    pid_seq = tl.program_id(2)
    
    # Calculate base pointers for this thread
    x_base = x_ptr + pid_batch * stride_x_batch + pid_head * stride_x_head + pid_seq * stride_x_seq
    out_base = out_ptr + pid_batch * stride_out_batch + pid_head * stride_out_head + pid_seq * stride_out_seq
    cos_base = cos_ptr + pid_seq * stride_cos_seq
    sin_base = sin_ptr + pid_seq * stride_sin_seq
    
    # Process head dimension in pairs (for complex rotation)
    for i in range(0, HEAD_DIM // 2):
        # Load x values (real and imaginary parts)
        x1 = tl.load(x_base + i * 2)
        x2 = tl.load(x_base + i * 2 + 1)
        
        # Load cos/sin values
        cos_val = tl.load(cos_base + i)
        sin_val = tl.load(sin_base + i)
        
        # Apply rotation: [x1, x2] -> [x1*cos + x2*sin, -x1*sin + x2*cos]
        y1 = x1 * cos_val + x2 * sin_val
        y2 = -x1 * sin_val + x2 * cos_val
        
        # Store results
        tl.store(out_base + i * 2, y1)
        tl.store(out_base + i * 2 + 1, y2)

class TritonRoPE:
    def __init__(self, dim: int, max_seq_len: int, base: float = 10000.0):
        """
        Triton-based RoPE implementation
        
        Args:
            dim: Head dimension (must be even)
            max_seq_len: Maximum sequence length
            base: Base for frequency calculation
        """
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        assert dim % 2 == 0, "Head dimension must be even for RoPE"
        
        # Precompute frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.einsum('i,j->ij', t, inv_freq)
        
        # Register as buffers
        self.register_buffer('cos', freqs.cos(), persistent=False)
        self.register_buffer('sin', freqs.sin(), persistent=False)
    
    def register_buffer(self, name: str, tensor: torch.Tensor, persistent: bool = True):
        """Simple buffer registration"""
        setattr(self, name, tensor)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RoPE using Triton kernel
        
        Args:
            x: Input tensor of shape [batch_size, n_heads, seq_len, head_dim]
        
        Returns:
            Output tensor with RoPE applied
        """
        batch_size, n_heads, seq_len, head_dim = x.shape
        
        # Ensure we have enough precomputed values
        if seq_len > self.cos.size(0):
            raise ValueError(f"Sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}")
        
        # Prepare output tensor
        output = torch.empty_like(x)
        
        # Get cos/sin for current sequence length
        cos_seq = self.cos[:seq_len].contiguous()  # [seq_len, head_dim//2]
        sin_seq = self.sin[:seq_len].contiguous()  # [seq_len, head_dim//2]
        
        # Launch Triton kernel
        grid = (batch_size, n_heads, seq_len)
        
        rope_forward_kernel[grid](
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

# PyTorch RoPE implementation (from your original code, slightly modified)
class PyTorchRoPE:
    def __init__(self, dim: int, max_seq_len: int, base: float = 10000.0):
        self.dim = dim
        self.max_seq_len = max_seq_len
        
        # Use the same frequency computation as original
        angular_freq = (1 / base) ** torch.linspace(0, 1, steps=dim//4, dtype=torch.float32)
        angular_freq = torch.cat([angular_freq, angular_freq.new_zeros(dim//4)])
        t = torch.arange(max_seq_len, dtype=torch.float32)
        theta = torch.einsum("i,j -> ij", t, angular_freq)
        
        self.register_buffer('cos', theta.cos(), persistent=False)
        self.register_buffer('sin', theta.sin(), persistent=False)
    
    def register_buffer(self, name: str, tensor: torch.Tensor, persistent: bool = True):
        setattr(self, name, tensor)
    
    def forward(self, x_BTHD: torch.Tensor) -> torch.Tensor:
        """Apply RoPE using PyTorch operations"""
        assert self.cos.size(0) >= x_BTHD.size(-3)
        cos = self.cos[None, :x_BTHD.size(-3), None, :]
        sin = self.sin[None, :x_BTHD.size(-3), None, :]
        
        x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), -1).type_as(x_BTHD)

def benchmark_rope_implementations():
    """Benchmark both RoPE implementations across different configurations"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu':
        print("Warning: CUDA not available, benchmarks will run on CPU")
        return
    
    # Test configurations
    configs = [
        # (batch_size, n_heads, seq_len, head_dim)
        (1, 8, 128, 64),
        (2, 8, 256, 64), 
        (4, 12, 512, 64),
        (8, 16, 1024, 64),
        (1, 8, 2048, 128),
        (2, 16, 512, 128),
    ]
    
    results = []
    
    print("🚀 Benchmarking RoPE Implementations")
    print("=" * 60)
    
    for batch_size, n_heads, seq_len, head_dim in configs:
        print(f"\nConfiguration: B={batch_size}, H={n_heads}, L={seq_len}, D={head_dim}")
        
        # Create test data
        x = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device, dtype=torch.float32)
        
        # Initialize implementations
        triton_rope = TritonRoPE(head_dim, max(seq_len, 4096))
        pytorch_rope = PyTorchRoPE(head_dim, max(seq_len, 4096))
        
        # Move buffers to device
        triton_rope.cos = triton_rope.cos.to(device)
        triton_rope.sin = triton_rope.sin.to(device)
        pytorch_rope.cos = pytorch_rope.cos.to(device)
        pytorch_rope.sin = pytorch_rope.sin.to(device)
        
        # Warmup
        for _ in range(10):
            _ = triton_rope.forward(x)
            _ = pytorch_rope.forward(x)
        
        torch.cuda.synchronize()
        
        # Benchmark Triton implementation
        n_runs = 100
        start_time = time.time()
        for _ in range(n_runs):
            triton_output = triton_rope.forward(x)
        torch.cuda.synchronize()
        triton_time = (time.time() - start_time) / n_runs
        
        # Benchmark PyTorch implementation  
        start_time = time.time()
        for _ in range(n_runs):
            pytorch_output = pytorch_rope.forward(x)
        torch.cuda.synchronize()
        pytorch_time = (time.time() - start_time) / n_runs
        
        # Verify correctness (approximately, due to different freq computations)
        max_diff = torch.max(torch.abs(triton_output - pytorch_output)).item()
        relative_error = max_diff / torch.max(torch.abs(pytorch_output)).item()
        
        speedup = pytorch_time / triton_time
        
        print(f"  Triton:  {triton_time*1000:.3f} ms")
        print(f"  PyTorch: {pytorch_time*1000:.3f} ms") 
        print(f"  Speedup: {speedup:.2f}x")
        print(f"  Max diff: {max_diff:.6f}")
        print(f"  Relative error: {relative_error:.6f}")
        
        results.append({
            'config': f"{batch_size}x{n_heads}x{seq_len}x{head_dim}",
            'batch_size': batch_size,
            'n_heads': n_heads, 
            'seq_len': seq_len,
            'head_dim': head_dim,
            'triton_time': triton_time * 1000,
            'pytorch_time': pytorch_time * 1000,
            'speedup': speedup,
            'max_diff': max_diff,
            'relative_error': relative_error
        })
    
    return results

def plot_benchmark_results(results):
    """Plot benchmark results"""
    if not results:
        return
        
    configs = [r['config'] for r in results]
    triton_times = [r['triton_time'] for r in results]
    pytorch_times = [r['pytorch_time'] for r in results]
    speedups = [r['speedup'] for r in results]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Runtime comparison
    x_pos = np.arange(len(configs))
    width = 0.35
    
    ax1.bar(x_pos - width/2, triton_times, width, label='Triton', alpha=0.8)
    ax1.bar(x_pos + width/2, pytorch_times, width, label='PyTorch', alpha=0.8)
    
    ax1.set_xlabel('Configuration (BxHxLxD)')
    ax1.set_ylabel('Time (ms)')
    ax1.set_title('RoPE Implementation Runtime Comparison')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(configs, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Speedup plot
    colors = ['green' if s > 1 else 'red' for s in speedups]
    bars = ax2.bar(x_pos, speedups, color=colors, alpha=0.7)
    ax2.axhline(y=1, color='black', linestyle='--', alpha=0.5)
    
    ax2.set_xlabel('Configuration (BxHxLxD)')
    ax2.set_ylabel('Speedup (PyTorch/Triton)')
    ax2.set_title('Triton RoPE Speedup vs PyTorch')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(configs, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    
    # Add speedup labels on bars
    for bar, speedup in zip(bars, speedups):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{speedup:.2f}x', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()

def memory_usage_analysis():
    """Analyze memory usage of both implementations"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu':
        return
        
    print("\n🔍 Memory Usage Analysis")
    print("=" * 40)
    
    config = (4, 16, 1024, 128)  # Large config for memory analysis
    batch_size, n_heads, seq_len, head_dim = config
    
    x = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device)
    
    # Triton implementation
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    triton_rope = TritonRoPE(head_dim, seq_len + 100)
    triton_rope.cos = triton_rope.cos.to(device)
    triton_rope.sin = triton_rope.sin.to(device)
    
    _ = triton_rope.forward(x)
    triton_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
    
    # PyTorch implementation
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    pytorch_rope = PyTorchRoPE(head_dim, seq_len + 100)
    pytorch_rope.cos = pytorch_rope.cos.to(device)
    pytorch_rope.sin = pytorch_rope.sin.to(device)
    
    _ = pytorch_rope.forward(x)
    pytorch_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
    
    print(f"Configuration: {batch_size}x{n_heads}x{seq_len}x{head_dim}")
    print(f"Triton peak memory:  {triton_memory:.2f} MB")
    print(f"PyTorch peak memory: {pytorch_memory:.2f} MB")
    print(f"Memory ratio: {pytorch_memory/triton_memory:.2f}x")

if __name__ == "__main__":
    # Check if we have GPU support
    if not torch.cuda.is_available():
        print("❌ CUDA not available. This benchmark requires GPU support.")
        exit(1)
        
    print(f"🔍 Device: {torch.cuda.get_device_name()}")
    print(f"💾 Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Run benchmarks
    results = benchmark_rope_implementations()
    
    if results:
        print("\n📊 Summary Results:")
        print("=" * 60)
        for result in results:
            print(f"{result['config']:>20}: {result['speedup']:>6.2f}x speedup, "
                  f"{result['relative_error']:.2e} rel_error")
        
        # Plot results
        plot_benchmark_results(results)
        
        # Memory analysis
        memory_usage_analysis()
        
        # Summary statistics
        speedups = [r['speedup'] for r in results]
        print(f"\n🏆 Performance Summary:")
        print(f"   Average speedup: {np.mean(speedups):.2f}x")
        print(f"   Best speedup: {np.max(speedups):.2f}x")
        print(f"   Worst speedup: {np.min(speedups):.2f}x")
    else:
        print("❌ No benchmark results generated")
