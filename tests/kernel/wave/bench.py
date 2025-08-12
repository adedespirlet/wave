import itertools
from typing import Optional, Tuple, Union

import torch
import triton
import triton.language as tl
from torch import nn


import wave_lang.kernel.lang as tkl
import wave_lang.kernel.wave as tkw
from wave_lang.kernel.lang.global_symbols import *
from wave_lang.kernel.wave.utils.run_utils import (
    set_default_run_config,
)
from wave_lang.kernel.wave.utils.torch_utils import (
    device_zeros,
)
from wave_lang.kernel.wave.compile import WaveCompileOptions, wave_compile


@triton.jit
def fused_rmsnorm_kernel(
    output_ptr,
    activ_ptr,
    weight_ptr,
    eps: tl.constexpr,
    hidden_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    input_start = pid * hidden_dim

    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < hidden_dim

    a_ = tl.load(activ_ptr + input_start + offsets, mask=mask, other=0.0)
    a = a_.to(tl.float32)
    rms = tl.sqrt(tl.sum(a * a, axis=0) / hidden_dim + eps)

    w1_ = tl.load(weight_ptr + offsets, mask=mask, other=0.0)
    w1 = w1_.to(tl.float32)

    a_rms = a / rms * w1

    tl.store(
        output_ptr + input_start + offsets,
        a_rms,  # implicitly casts to output dtype here
        mask=mask,
    )


def fused_rmsnorm(x, weight, eps: float = 1e-6, autotune=False, inplace=False):
    assert len(x.shape) == 2
    if inplace:
        output = x
    else:
        output = torch.empty_like(x)
    bs, hidden_dim = x.shape
    max_warps = 16
    config = {
        "BLOCK_SIZE": triton.next_power_of_2(hidden_dim),
        "num_warps": max(
            min(triton.next_power_of_2(triton.cdiv(hidden_dim, 256)), max_warps), 4
        ),
    }
 
    compiled_kernel = fused_rmsnorm_kernel[(bs,)](
        output, x, weight, eps=eps, hidden_dim=hidden_dim, **config
    )
    # print(compiled_kernel.asm['amdgcn'])
    return output


def get_rmsnorm_wave(shape, eps: float = 1e-6):
    
    override_mlir_str128 = """
    #map = affine_map<()[s0, s1] -> ((s0 mod 64) * 8 + s1 * 512)>
    #map1 = affine_map<()[s0, s1] -> (s0 floordiv 64 + s1 * 4 )>
    #map2 = affine_map<()[s0] -> (s0 floordiv 64 )>
    #translation = #iree_codegen.translation_info<pipeline = None workgroup_size = [256, 1, 1] subgroup_size = 64>
    module attributes {transform.with_named_sequence} {
    stream.executable private @test {
        stream.executable.export public @test workgroups() -> (index, index, index) {
        %c1 = arith.constant 1 : index
        %c2 = arith.constant 32 : index
        stream.return %c1, %c2, %c1 : index, index, index
        }
        builtin.module {
        func.func @rmsnorm(%arg0: !stream.binding, %arg1: !stream.binding, %arg2: !stream.binding) attributes {translation_info = #translation} {
            %cst = arith.constant dense<5.120000e+03> : vector<1xf16>
            %c32_i32 = arith.constant 32 : i32
            %c0_i32 = arith.constant 0 : i32
            %c16_i32 = arith.constant 16 : i32
            %c8_i32 = arith.constant 8 : i32
            %c4_i32 = arith.constant 4 : i32
            %c2_i32 = arith.constant 2 : i32
            %c64_i32 = arith.constant 64 : i32
            %c1_i32 = arith.constant 1 : i32
            %c1 = arith.constant 1 : index
            %c0 = arith.constant 0 : index
            %c10 = arith.constant 10 : index
            %cst_0 = arith.constant dense<0.000000e+00> : vector<1xf16>
            %block_id_y = gpu.block_id  y upper_bound 32
            %thread_id_x = gpu.thread_id  x upper_bound 256
            %alloc = memref.alloc() : memref<40960xi8, #gpu.address_space<workgroup>>
            %buffer = memref.view %alloc[%c0][] : memref<40960xi8, #gpu.address_space<workgroup>> to memref<4x5120xf16, #gpu.address_space<workgroup>>
            %input = stream.binding.subspan %arg0[%c0] : !stream.binding -> memref<128x5120xf16, strided<[5120, 1], offset: ?>>
            %1 = scf.for %arg3 = %c0 to %c10 step %c1 iter_args(%arg4 = %cst_0) -> (vector<1xf16>) {
            %row = affine.apply #map1()[%thread_id_x,%block_id_y]
            %rowbuffer = affine.apply #map2()[%thread_id_x]
            %col = affine.apply #map()[%thread_id_x, %arg3]
            %val = vector.load %input[%row, %col] : memref<128x5120xf16, strided<[5120, 1], offset: ?>>, vector<8xf16>
            vector.store %val, %buffer[%rowbuffer, %col] : memref<4x5120xf16, #gpu.address_space<workgroup>>, vector<8xf16>
            %15 = arith.mulf %val, %val : vector<8xf16>
            %30 = vector.reduction <add>, %15 : vector<8xf16> into f16
            %r = gpu.subgroup_reduce add %30 : (f16) -> f16
            %31 = vector.broadcast %r : f16 to vector<1xf16>
            %38 = arith.addf %31, %arg4 : vector<1xf16>
            scf.yield %38 : vector<1xf16>
            }
            %2 = arith.divf %1, %cst : vector<1xf16>
            %3 = math.sqrt %2 : vector<1xf16>
            %broadcasted = vector.broadcast %3 : vector<1xf16> to vector<8xf16>
            %185 = stream.binding.subspan %arg1[%c0] : !stream.binding -> memref<5120xf16, strided<[1], offset: ?>>
            %70 = stream.binding.subspan %arg2[%c0] : !stream.binding -> memref<128x5120xf16, strided<[5120, 1], offset: ?>>
            scf.for %arg3 = %c0 to %c10 step %c1 {
            %rowbuffer = affine.apply #map2()[%thread_id_x]
            %row = affine.apply #map1()[%thread_id_x,%block_id_y]
            %col = affine.apply #map()[%thread_id_x, %arg3]
            %val = vector.load %buffer[%rowbuffer, %col] : memref<4x5120xf16, #gpu.address_space<workgroup>>, vector<8xf16>
            %53 = vector.load %185[%col] : memref<5120xf16, strided<[1], offset: ?>>, vector<8xf16>
            %54 = arith.divf %val, %broadcasted : vector<8xf16>
            %55 = arith.mulf %54, %53 : vector<8xf16>
            vector.store %55, %70[%row, %col] : memref<128x5120xf16, strided<[5120, 1], offset: ?>>, vector<8xf16>
            }
            return
        }
        }
    }
    func.func @isolated_benchmark$async(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view, %arg2: !hal.buffer_view, %arg3: !hal.fence, %arg4: !hal.fence) -> !hal.buffer_view {
        %0 = hal.tensor.import wait(%arg3) => %arg0 : !hal.buffer_view -> tensor<128x5120xf16>
        %1 = hal.tensor.import wait(%arg3) => %arg1 : !hal.buffer_view -> tensor<5120xf16>
        %2 = hal.tensor.import wait(%arg3) => %arg2 : !hal.buffer_view -> tensor<128x5120xf16>
        %3 = flow.dispatch @rmsnorm::@rmsnorm(%0, %1, %2) : (tensor<128x5120xf16>, tensor<5120xf16>, tensor<128x5120xf16>) -> %2
        %4 = hal.tensor.barrier join(%3 : tensor<128x5120xf16>) => %arg4 : !hal.fence
        %5 = hal.tensor.export %4 : tensor<128x5120xf16> -> !hal.buffer_view
        return %5 : !hal.buffer_view
    }
    }
    """
    M = tkl.sym.M
    N = tkl.sym.N
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
    
    constraints: list[tkw.Constraint] = [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            vector_shapes={M: 1, N: BLOCK_N },
        )
    ]
    constraints += [tkw.WorkgroupConstraint(M, BLOCK_M, 1)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 0)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N)]

    @tkw.wave(constraints)
    def rmsnorm(
        a: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
        weight: tkl.Memory[N, ADDRESS_SPACE, tkl.f16],
        c: tkl.Memory[M, N, ADDRESS_SPACE, tkl.f16],
    ):
        length_embedding = tkl.Register[M, tkl.f32](N)
        eps_reg = tkl.Register[M, tkl.f32](eps)
        a_reg = tkw.read(a)
        a_reg = tkw.cast(a_reg, tkl.f32)
        mean = tkw.sum(a_reg * a_reg, dim=N, block=False) / length_embedding + eps_reg
        rms = tkw.sqrt(mean)
        rms_broad = tkw.broadcast(rms, [M, N])
        a_scaled = a_reg * rms_broad
        w_reg = tkw.read(weight)
        w_reg = tkw.cast(w_reg, tkl.f32)
        w_broad = tkw.broadcast(w_reg, [M, N])
        output = a_scaled * w_broad
        output = tkw.cast(output, tkl.f16)
        tkw.write(output, c)

    options = WaveCompileOptions(
        subs={
            M: shape[0],
            N: shape[1],
            BLOCK_M: 1,
            BLOCK_N: shape[1],
            ADDRESS_SPACE: GLOBAL_ADDRESS_SPACE,
        },
        canonicalize=True,
        use_buffer_load_ops=True,
        use_buffer_store_ops=True,
        #wave_runtime=True,
        #override_mlir=override_mlir_str128,
    )
    options = set_default_run_config(options)
    return wave_compile(options, rmsnorm)

def rmsnorm_wave(
    kernel,
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
):
    c = torch.empty_like(x)
    kernel(x, weight, c)

    return c


class HuggingFaceRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        orig_dtype = x.dtype
        x = x.to(torch.float32)
        if residual is not None:
            x = x + residual.to(torch.float32)
            residual = x.to(orig_dtype)

        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        x = x.to(orig_dtype) * self.weight
        if residual is None:
            return x
        else:
            return x, residual


def rmsnorm_naive(
    x: torch.Tensor,
    weight: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
):
    naive_norm = HuggingFaceRMSNorm(x.shape[-1], eps=eps)
    naive_norm.weight = nn.Parameter(weight)
    naive_norm = naive_norm.to(x.device)

    orig_shape = x.shape
    x = x.view(-1, x.shape[-1])
    if residual is not None:
        residual = residual.view(-1, residual.shape[-1])

    output = naive_norm(x, residual)

    if isinstance(output, tuple):
        output = (output[0].view(orig_shape), output[1].view(orig_shape))
    else:
        output = output.view(orig_shape)
    return output


def calculate_diff(batch_size, seq_len, hidden_size, use_residual=True):
    dtype = torch.float16
    x = torch.randn(seq_len, hidden_size, dtype=dtype, device="cuda")
    weight = torch.ones(hidden_size, dtype=dtype, device="cuda")
    residual = torch.randn_like(x) if use_residual else None
    wave_kernel = get_rmsnorm_wave(x.shape)

    output_naive = rmsnorm_naive(
        x.clone(), weight, residual.clone() if residual is not None else None
    )
    output_wave = rmsnorm_wave(
        wave_kernel,
        x.clone(),
        weight,
        # residual.clone() if residual is not None else None,
    )
    output_triton = fused_rmsnorm(
        x.clone(), weight
    )

    if use_residual:
        output_naive = output_naive[0]
        output_wave = output_wave[0]
        output_triton = output_triton[0]

    print(f"Naive output={output_naive}")
    print(f"Wave output={output_wave}")
    print(f"Triton output={output_triton}")

    if torch.allclose(
        output_naive, output_wave, atol=1e-2, rtol=1e-2
    ) and torch.allclose(output_naive, output_triton, atol=1e-2, rtol=1e-2):
        print("✅ All implementations match")
    else:
        print("❌ Implementations differ")


batch_size_range = [1]  # [2**i for i in range(0, 7, 2)]
# seq_length_range = [1, 16] + [2**i for i in range(6, 11, 1)]
# hidden_size_range = [i * 128 for i in [1, 32, 48]] + [5120]

seq_length_range = [128]
hidden_size_range = [5120]
configs = list(itertools.product(batch_size_range, seq_length_range, hidden_size_range))


def get_benchmark(use_residual):
    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["batch_size", "seq_len", "hidden_size",],
            x_vals=[list(_) for _ in configs],
            line_arg="provider",
            line_vals=["huggingface", "wave", "triton"],
            line_names=["HuggingFace", "Wave", "Triton"],
            styles=[("blue", "-"), ("red", "-"), ("green", "-"), ("orange", "-")],
            ylabel="us",
            plot_name=f"rmsnorm-performance-{'with' if use_residual else 'without'}-residual",
            args={},
        )
    )
    def benchmark(batch_size, seq_len, hidden_size, provider):
        dtype = torch.float16

        x = torch.randn(seq_len, hidden_size, dtype=dtype, device="cuda")
        weight = torch.ones(hidden_size, dtype=dtype, device="cuda")
        residual = torch.randn_like(x) if use_residual else None

        quantiles = [0.5, 0.2, 0.8]

        if provider == "huggingface":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: rmsnorm_naive(
                    x.clone(),
                    weight,
                    residual.clone() if residual is not None else None,
                ),
                quantiles=quantiles,
            )
        elif provider == "wave":
            wave_kernel = get_rmsnorm_wave(x.shape)
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: rmsnorm_wave(
                    wave_kernel,
                    x.clone(),
                    weight
                ),
                quantiles=quantiles,
            )
        elif provider == "triton":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: fused_rmsnorm(
                    x.clone(),
                    weight,
                ),
                quantiles=quantiles,
            )
        else:
            print("no provider")

        return 1000 * ms, 1000 * max_ms, 1000 * min_ms

    return benchmark


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use_residual", action="store_true", help="Whether to use residual connection"
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="./results",
        help="Path to save rmsnorm benchmark results",
    )
    args = parser.parse_args()

    # Run correctness test
    calculate_diff(
        batch_size=1, seq_len=128, hidden_size=5120, use_residual=args.use_residual
    )

    # Get the benchmark function with proper use_residual setting
    benchmark = get_benchmark(args.use_residual)
    # Run performance benchmark
    benchmark.run(print_data=True, save_path=args.save_path)