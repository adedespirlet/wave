import triton
import triton.language as tl
import torch
import itertools
import math

import triton.compiler as tc
from .torch_kernels import moe_align_block_size_pytorch
from pathlib import Path
import datetime as dt


@triton.jit
def _moe_gemm_kernel_triton(
    a_ptr,              # (M, K)
    b_ptr,              # (E, N, K)
    c2d_ptr,            # (M*topk, N)  -- 2D view of C to match offs_token indexing
    sorted_token_ids_ptr,   # (EM,)
    expert_ids_ptr,         # (num_blocks_m,)
    num_tokens_post_padded_ptr,  # (1,)

    # dimensions
    N, K, EM, num_valid_tokens,

    # strides
    stride_am, stride_ak,     # A
    stride_be, stride_bk, stride_bn,  # B  (E,N,K) with layout E-major
    stride_c2dm, stride_c2dn, # C2D  (M*topk, N)

    # meta
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,     # tl.float32 (accumulator)
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # early-out if padded rows
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return

    # per-block token slice
    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id).to(tl.int64)
    token_mask = offs_token < num_valid_tokens

    # expert for this M-block
    off_expert = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    # if -1, nothing to compute (not in this expert-parallel shard)
    if off_expert == -1:
        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=compute_type)
        offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c2d_ptr + offs_token[:, None] * stride_c2dm + offs_n[None, :] * stride_c2dn
        c_mask = token_mask[:, None] & (offs_n[None, :] < N)
        tl.store(c_ptrs, acc, mask=c_mask)
        return

    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # A pointers (flattened token pairs // top_k → original token)
    a_ptrs = a_ptr + ( (offs_token[:, None] // top_k) * stride_am + offs_k[None, :] * stride_ak )

    # B pointers: expert-major, NxK per expert
    b_ptrs = (
        b_ptr
        + off_expert * stride_be
        + (offs_k[:, None] * stride_bk + (offs_n[None, :] % N) * stride_bn)
    )

    # fp32 accumulation
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=compute_type)

    # K loop
    num_k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    for kt in range(0, num_k_tiles):
        # for last tile, mask loads if K not a multiple of BLOCK_SIZE_K
        k_left = K - kt * BLOCK_SIZE_K
        a = tl.load(a_ptrs, mask=token_mask[:, None] & (offs_k[None, :] < k_left), other=0)
        b = tl.load(b_ptrs,  mask=(offs_k[:, None] < k_left), other=0)
        acc += tl.dot(a, b)  # fp16/bf16→fp32

        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # write-back into (M*topk, N) view using offs_token as row index
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c2d_ptr + offs_token[:, None] * stride_c2dm + offs_n[None, :] * stride_c2dn
    c_mask = token_mask[:, None] & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


# --------- Python launcher that uses your test’s tensors ---------
def moe_gemm_triton(a, b, c, sorted_ids, expert_ids, num_tokens_post_padded, topk,
                    block_m=64, block_n=64, block_k=32, group_m=8):
    """
    a: (M, K)          dtype: fp16/bf16
    b: (E, N, K)       dtype: fp16/bf16 (same as a)
    c: (M, topk, N)    dtype: same as a (or fp16/bf16); accumulator is fp32
    sorted_ids: (EM,)  int32/int64
    expert_ids: (ceil(EM/block_m),) int32/int64  (one expert id per M-block)
    num_tokens_post_padded: (1,) int32/int64
    """
    assert a.device.type == "cuda" and b.device.type == "cuda" and c.device.type == "cuda"
    M, K = a.shape
    E, N, Kb = b.shape
    assert K == Kb, "B last dim must equal K"
    assert c.shape == (M, topk, N)
    EM = sorted_ids.numel()
    compute_type = tl.float32

    # convenience: make 2D view of c as (M*topk, N) to match offs_token addressing
    c2d = c.view(M * topk, N)

    grid = (triton.cdiv(EM, block_m) * triton.cdiv(N, block_n),)

    _moe_gemm_kernel_triton[grid](
        a, b, c2d,
        sorted_ids, expert_ids, num_tokens_post_padded,
        N, K, EM, M * topk,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(2), b.stride(1),
        c2d.stride(0), c2d.stride(1),
        BLOCK_SIZE_M=block_m,
        BLOCK_SIZE_N=block_n,
        BLOCK_SIZE_K=block_k,
        GROUP_SIZE_M=group_m,
        top_k=topk,
        compute_type=compute_type,
        num_warps=4,      # good defaults for 64x64x32 tiles on RDNA/CDNA
        num_stages=2,
    )


   
@torch.no_grad()
def build_sorted_ids_and_expert_blocks(topk_ids: torch.Tensor, num_experts: int, block_m: int):
    """
    Given topk_ids: (M, topk) of expert indices, build:
      - sorted_ids: (EM_padded,) int32 — token*topk+slot, grouped by expert, padded per-expert to multiples of block_m
      - expert_ids: (num_m_blocks,) int32 — expert id for each M-block (64 rows), -1 for host-padding blocks
      - num_tokens_post_padded: (1,) int32 — total length of sorted_ids after padding
    """
    device = topk_ids.device
    M, topk = topk_ids.shape
    torch.manual_seed(0)
    
    max_num_tokens_padded = topk_ids.numel() + (num_experts + 1) * (block_m - 1)
    sorted_ids = torch.empty(
        (max_num_tokens_padded,), dtype=torch.int32, device=topk_ids.device
    )
    max_num_m_blocks = -(max_num_tokens_padded // -block_m)
    expert_ids = torch.empty(
        (max_num_m_blocks,), dtype=torch.int32, device=topk_ids.device
    )
    num_tokens_post_pad = torch.empty((1), dtype=torch.int32, device=topk_ids.device)

    fuse_sorted_ids_padding = sorted_ids.shape[0] <= 4096
    if not fuse_sorted_ids_padding:
        sorted_ids.fill_(topk_ids.numel())
    
    # Populate using the same routine as the test harness
    moe_align_block_size_pytorch(
        topk_ids, num_experts, block_m, sorted_ids, expert_ids, num_tokens_post_pad
    )

    return sorted_ids, expert_ids, num_tokens_post_pad, max_num_tokens_padded
    


def compare_once(M, N, K):
    topk = 2
    num_experts = 8
    block_m, block_n, block_k = 64, 64, 32
    dtype = torch.float16
    device = torch.device("cuda")

    scores = torch.rand(M, num_experts, device=device)
    _, topk_ids = torch.topk(scores, k=topk, dim=1)

    sorted_ids, expert_ids, num_tokens_post_padded, max_num_tokens_padded = build_sorted_ids_and_expert_blocks(
        topk_ids, num_experts, block_m
    )
    
    #  a = torch.rand((num_tokens, k), dtype=dtype, device='cuda')
    # b = torch.rand((num_experts, n, k), dtype=dtype, device='cuda')
    # c = torch.zeros(num_tokens, topk, n, dtype=dtype, device='cuda')
    
    a = torch.rand((M, K), dtype=dtype, device=device)
    b = torch.rand((num_experts, N, K), dtype=dtype, device=device)

    c_tri = torch.zeros(M, topk, N, dtype=dtype, device="cuda")

    moe_gemm_triton(a, b, c_tri, sorted_ids, expert_ids, num_tokens_post_padded, topk,
                    block_m=block_m, block_n=block_n, block_k=block_k, group_m=8)
    

    
    rtol, atol = 1e-1, 1e-2


    # print slice + diffs
    def show_outputs(ref, tri,wave, rows=4, cols=16):
        torch.set_printoptions(sci_mode=False, linewidth=200, precision=4)
        print("\n=== sanity slice [0:%d, 0:2, 0:%d] ===" % (rows, cols))
      
        print("--- c_triton ---")
        print(tri[:rows, :2, :cols])

    
    
    

if __name__ == "__main__":
    import torch

    # Reproducibility + device sanity
    torch.manual_seed(0)
    assert torch.cuda.is_available(), "No HIP/CUDA device visible"

    # Match your test case
    #M, N, K = 33, 256, 128
    
    
    #M,N,K= 2048,1024,256
    M,N,K= 16384,32768,6144

    compare_once(M, N, K)

  