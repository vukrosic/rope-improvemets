import torch
import triton
import triton.language as tl
import math


@triton.jit
def rope_kernel(
    x_ptr,
    cos_ptr,
    sin_ptr,
    output_ptr,
    batch_size,
    n_heads,
    seq_len,
    d_k,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for Rotary Position Embedding (RoPE).
    
    Args:
        x_ptr: Input tensor pointer [batch_size, n_heads, seq_len, d_k]
        cos_ptr: Cosine values pointer [seq_len, d_k//2]
        sin_ptr: Sine values pointer [seq_len, d_k//2]
        output_ptr: Output tensor pointer [batch_size, n_heads, seq_len, d_k]
    """
    # Get program ID for parallelization
    pid = tl.program_id(axis=0)
    
    # Calculate which batch, head, and sequence position this thread handles
    total_seq_positions = batch_size * n_heads * seq_len
    
    if pid >= total_seq_positions:
        return
    
    # Decompose the linear index
    batch_idx = pid // (n_heads * seq_len)
    remaining = pid % (n_heads * seq_len)
    head_idx = remaining // seq_len
    seq_idx = remaining % seq_len
    
    # Calculate half dimension
    half_d_k = d_k // 2
    
    # Base offset for this specific position in the tensor
    base_offset = batch_idx * (n_heads * seq_len * d_k) + head_idx * (seq_len * d_k) + seq_idx * d_k
    
    # Load cos and sin values for this sequence position
    cos_sin_offset = seq_idx * half_d_k
    
    # Process in blocks to handle different d_k sizes
    for block_start in range(0, half_d_k, BLOCK_SIZE):
        block_end = min(block_start + BLOCK_SIZE, half_d_k)
        block_size = block_end - block_start
        
        if block_size <= 0:
            break
            
        # Create masks for this block
        block_mask = tl.arange(0, BLOCK_SIZE) < block_size
        
        # Load cos and sin values
        cos_offset = cos_sin_offset + block_start + tl.arange(0, BLOCK_SIZE)
        sin_offset = cos_sin_offset + block_start + tl.arange(0, BLOCK_SIZE)
        
        cos_vals = tl.load(cos_ptr + cos_offset, mask=block_mask, other=0.0)
        sin_vals = tl.load(sin_ptr + sin_offset, mask=block_mask, other=0.0)
        
        # Load first half (x1) and second half (x2) of the input
        x1_offset = base_offset + block_start + tl.arange(0, BLOCK_SIZE)
        x2_offset = base_offset + half_d_k + block_start + tl.arange(0, BLOCK_SIZE)
        
        x1 = tl.load(x_ptr + x1_offset, mask=block_mask, other=0.0)
        x2 = tl.load(x_ptr + x2_offset, mask=block_mask, other=0.0)
        
        # Apply RoPE transformation
        # y1 = x1 * cos + x2 * sin
        # y2 = x1 * (-sin) + x2 * cos
        y1 = x1 * cos_vals + x2 * sin_vals
        y2 = x1 * (-sin_vals) + x2 * cos_vals
        
        # Store results
        y1_offset = base_offset + block_start + tl.arange(0, BLOCK_SIZE)
        y2_offset = base_offset + half_d_k + block_start + tl.arange(0, BLOCK_SIZE)
        
        tl.store(output_ptr + y1_offset, y1, mask=block_mask)
        tl.store(output_ptr + y2_offset, y2, mask=block_mask)


class TritonRoPE(torch.nn.Module):
    """
    Optimized Rotary Position Embedding using Triton kernels.
    Drop-in replacement for the original Rotary class.
    """
    
    def __init__(self, dim: int, max_seq_len: int):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        
        # Precompute angular frequencies
        angular_freq = (1 / 10000) ** torch.linspace(0, 1, steps=dim//4, dtype=torch.float32)
        angular_freq = torch.cat([angular_freq, angular_freq.new_zeros(dim//4)])
        t = torch.arange(max_seq_len, dtype=torch.float32)
        theta = torch.einsum("i,j -> ij", t, angular_freq)
        
        self.register_buffer('cos', theta.cos(), persistent=False)
        self.register_buffer('sin', theta.sin(), persistent=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RoPE to input tensor.
        
        Args:
            x: Input tensor of shape [batch_size, n_heads, seq_len, d_k]
            
        Returns:
            Output tensor with RoPE applied, same shape as input
        """
        batch_size, n_heads, seq_len, d_k = x.shape
        
        # Ensure we don't exceed precomputed sequence length
        assert seq_len <= self.max_seq_len, f"Sequence length {seq_len} exceeds max {self.max_seq_len}"
        assert d_k == self.dim, f"Head dimension {d_k} doesn't match expected {self.dim}"
        
        # Get relevant cos/sin values for current sequence length
        cos_vals = self.cos[:seq_len, :]  # [seq_len, d_k//2]
        sin_vals = self.sin[:seq_len, :]  # [seq_len, d_k//2]
        
        # Prepare output tensor
        output = torch.empty_like(x)
        
        # Calculate grid size and block size
        total_elements = batch_size * n_heads * seq_len
        BLOCK_SIZE = min(triton.next_power_of_2(d_k // 2), 1024)  # Triton limitation
        
        # Launch kernel
        grid = (total_elements,)
        
        rope_kernel[grid](
            x.contiguous(),
            cos_vals.contiguous(),
            sin_vals.contiguous(),
            output,
            batch_size,
            n_heads,
            seq_len,
            d_k,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        
        return output


def benchmark_rope_implementations(batch_size=8, n_heads=8, seq_len=512, d_k=64, num_warmup=10, num_trials=100):
    """
    Benchmark the original PyTorch RoPE vs Triton RoPE implementation.
    """
    from llm import Rotary  # Import original implementation
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if device.type == 'cpu':
        print("⚠️  CUDA not available, Triton kernels require GPU")
        return
    
    print(f"🔥 Benchmarking RoPE implementations on {device}")
    print(f"   Shape: [{batch_size}, {n_heads}, {seq_len}, {d_k}]")
    
    # Create test data
    x = torch.randn(batch_size, n_heads, seq_len, d_k, device=device, dtype=torch.float32)
    
    # Initialize both implementations
    original_rope = Rotary(d_k, seq_len * 2).to(device)
    triton_rope = TritonRoPE(d_k, seq_len * 2).to(device)
    
    # Warmup
    print("🔄 Warming up...")
    for _ in range(num_warmup):
        _ = original_rope(x)
        _ = triton_rope(x)
    
    torch.cuda.synchronize()
    
    # Benchmark original implementation
    print("📊 Benchmarking Original PyTorch RoPE...")
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)
    
    start_time.record()
    for _ in range(num_trials):
        original_output = original_rope(x)
    end_time.record()
    
    torch.cuda.synchronize()
    original_time = start_time.elapsed_time(end_time) / num_trials
    
    # Benchmark Triton implementation
    print("📊 Benchmarking Triton RoPE...")
    start_time.record()
    for _ in range(num_trials):
        triton_output = triton_rope(x)
    end_time.record()
    
    torch.cuda.synchronize()
    triton_time = start_time.elapsed_time(end_time) / num_trials
    
    # Verify correctness
    print("🔍 Verifying correctness...")
    max_diff = torch.max(torch.abs(original_output - triton_output)).item()
    mean_diff = torch.mean(torch.abs(original_output - triton_output)).item()
    
    # Results
    speedup = original_time / triton_time
    print(f"\n🏆 Results:")
    print(f"   Original PyTorch: {original_time:.3f} ms")
    print(f"   Triton Kernel:    {triton_time:.3f} ms")
    print(f"   Speedup:          {speedup:.2f}x")
    print(f"   Max difference:   {max_diff:.2e}")
    print(f"   Mean difference:  {mean_diff:.2e}")
    
    if max_diff < 1e-5:
        print("✅ Correctness check passed!")
    else:
        print("❌ Correctness check failed!")
    
    return speedup, max_diff


if __name__ == "__main__":
    # Run benchmark
    benchmark_rope_implementations()