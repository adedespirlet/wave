"""
GEMM Scheduling Part 2: Advanced Scheduling with Reordering and Staggering Waves

This example demonstrates advanced scheduling techniques for optimizing GPU performance:
1. Partitioning operations by dimensions to interleave compute with memory ops
2. Creating instruction clusters for optimal ordering
3. Wave priority manipulation (SetWavePrio) to prioritize compute waves over memory waves
4. Staggering waves for better overlap of computation and memory access
5. Scheduling barriers for fine-grained control
"""

import torch

import wave_lang.kernel.wave as tkw
import wave_lang.kernel.lang as tkl
import wave_lang.kernel.wave.wave_schedule as wave_schedule
from wave_lang.kernel.lang.global_symbols import *
from wave_lang.kernel.lang.wave_types import *
from wave_lang.kernel.wave.compile import WaveCompileOptions, wave_compile
from wave_lang.kernel.wave.utils.run_utils import set_default_run_config
from wave_lang.kernel.wave.scheduling.schedule import SchedulingType

from utils import parse_args, list_tests, run_test


def test_gemm_advanced_scheduling(is_debug=False):
    """
    Advanced GEMM scheduling with cluster reordering and ping-pong buffering.

    This example demonstrates sophisticated scheduling techniques:

    1. Partitioning: Split MMA operations by K dimension to interleave compute with memory ops
    2. Clustering: Group instructions for optimal execution order
    3. Wave Priority: Use SetWavePrio to prioritize compute waves over memory waves
    4. Ping-Pong (Stagger): Double-buffer shared memory to avoid stalls
    5. Barriers: Insert explicit synchronization points

    The resulting schedule maximizes instruction-level parallelism and hides memory latency
    by carefully interleaving memory operations with compute operations from different
    iterations.
    """
    # shape: tuple[int, int, int] = (128, 256, 1024)
    shape: tuple[int, int, int] = (40960, 256, 512)
    mfma_variant: tkw.MMAType = tkw.MMAType.F32_16x16x32_F16

    # Symbol definitions
    M = tkl.sym.M
    N = tkl.sym.N
    K = tkl.sym.K
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N
    BLOCK_K = tkl.sym.BLOCK_K
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
    ADDRESS_SPACE_0 = tkl.sym.ADDRESS_SPACE_0

    # Basic constraints needed for compilation
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.TilingConstraint(K, BLOCK_K)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 4)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 2)]

    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            mma_type=mfma_variant,
        )
    ]

    # Define the kernel (same as before, but we'll apply advanced scheduling)
    @tkw.wave(constraints)
    def gemm_prefetch(
        a: tkl.Memory[M, K, ADDRESS_SPACE, tkl.f16],
        b: tkl.Memory[N, K, ADDRESS_SPACE, tkl.f16],
        c: tkl.Memory[M, N, ADDRESS_SPACE_0, tkl.f32],
    ):
        c_reg = tkl.Register[M, N, tkl.f32](0.0)

        @tkw.iterate(K, init_args=[c_reg], tag="k_loop")
        def repeat(
            acc: tkl.Register[M, N, tkl.f32],
        ) -> tkl.Register[M, N, tkl.f32]:
            a_reg = tkw.read(a, tag="read_a")
            b_reg = tkw.read(b, tag="read_b")
            acc = tkw.mma(a_reg, b_reg, acc, tag="mma")
            return acc

        tkw.write(repeat, c)

    # Define the advanced schedule
    @wave_schedule.wave_schedule()
    def advanced_schedule():
        """
        Advanced scheduling with cluster-based reordering and ping-pong buffering.

        The schedule creates a sophisticated instruction ordering that:
        1. Interleaves compute (MMA) with memory operations
        2. Uses wave priorities to ensure compute waves get resources when needed
        3. Implements ping-pong buffering via stagger() for double buffering
        4. Carefully places barriers to ensure correctness while maximizing parallelism
        """
        # Get nodes to be manipulated in the schedule.
        k_loop = tkw.get_node_by_tag("k_loop")
        load_a = tkw.get_node_by_tag_and_type("read_a", tkw.Read)
        global_load_a, shared_load_a = tkw.partition_by_address_space(
            load_a, GLOBAL_ADDRESS_SPACE
        )
        shared_write_a = tkw.get_node_by_tag_and_type("read_a", tkw.Write)
        load_b = tkw.get_node_by_tag_and_type("read_b", tkw.Read)
        global_load_b, shared_load_b = tkw.partition_by_address_space(
            load_b, GLOBAL_ADDRESS_SPACE
        )
        shared_write_b = tkw.get_node_by_tag_and_type("read_b", tkw.Write)
        mma = tkw.get_node_by_tag("mma")

        pipeline_loop = tkw.pipeline(k_loop)
        # First, create the basic 2-stage pipeline
        with pipeline_loop as pl:
            pl.set_stage(
                [
                    (global_load_a, global_load_b),
                    (shared_write_a, shared_write_b),
                ],
            )
            pl.set_stage(
                [
                    (shared_load_a, shared_load_b),
                    (mma,),
                ],
            )

        # Now apply advanced scheduling to the KERNEL stage
        global_load_a = tkw.filter_nodes(global_load_a, subgraph=pipeline_loop.KERNEL)
        shared_load_a = tkw.filter_nodes(shared_load_a, subgraph=pipeline_loop.KERNEL)
        shared_write_a = tkw.filter_nodes(shared_write_a, subgraph=pipeline_loop.KERNEL)
        global_load_b = tkw.filter_nodes(global_load_b, subgraph=pipeline_loop.KERNEL)
        shared_load_b = tkw.filter_nodes(shared_load_b, subgraph=pipeline_loop.KERNEL)
        shared_write_b = tkw.filter_nodes(shared_write_b, subgraph=pipeline_loop.KERNEL)
        mma = tkw.filter_nodes(mma, subgraph=pipeline_loop.KERNEL)

        # Partition MMA operations by K dimension into 2 groups
        # This allows us to interleave the first half of MMA with prefetch for next iteration
        mma_0, mma_1 = tkw.partition_by_dim(mma, dim=K, num_partitions=2)

        # Similarly partition the shared memory loads
        shared_load_a_0, shared_load_a_1 = tkw.partition_by_dim(
            shared_load_a, dim=K, num_partitions=2
        )
        shared_load_b_0, shared_load_b_1 = tkw.partition_by_dim(
            shared_load_b, dim=K, num_partitions=2
        )

        # Create instruction clusters that define the execution order
        # Each cluster groups instructions that should execute together
        clusters = [
            # Cluster 1: First half of loads + prefetch for next iteration
            tkw.cluster(
                [
                    shared_load_a_0,  # Load first half of A from shared memory
                    shared_load_b_0,  # Load first half of B from shared memory
                    tkw.SchedulingBarrier([]),  # Barrier for scheduling control
                    global_load_a,  # Prefetch A for next iteration (overlapped!)
                    tkw.SchedulingBarrier([]),
                    shared_load_a_1,  # Load second half of A
                    shared_load_b_1,  # Load second half of B
                    tkw.SchedulingBarrier([]),
                    global_load_b,  # Prefetch B for next iteration (overlapped!)
                    tkw.WorkgroupBarrier(),  # Ensure all waves complete loads
                    tkw.SchedulingBarrier([]),
                ],
            ),
            # Cluster 2: First half of MMA operations with high priority
            tkw.cluster(
                [
                    tkw.SetWavePrio(1),  # Increase priority for compute
                    mma_0,  # Execute first half of MMA operations
                    tkw.SetWavePrio(0),  # Reset priority
                    tkw.SharedMemoryBarrier(),  # Sync shared memory
                    tkw.SchedulingBarrier([]),
                ],
            ),
            # Cluster 3: Write prefetched data to shared memory
            tkw.cluster(
                [
                    shared_write_a,  # Write prefetched A to shared memory
                    shared_write_b,  # Write prefetched B to shared memory
                    tkw.WorkgroupBarrier(),  # Ensure writes complete
                    tkw.SchedulingBarrier([]),
                ],
            ),
            # Cluster 4: Second half of MMA operations
            tkw.cluster(
                [
                    tkw.SetWavePrio(1),  # Increase priority for compute
                    mma_1,  # Execute second half of MMA operations
                    tkw.SetWavePrio(0),  # Reset priority
                    tkw.SchedulingBarrier([]),
                ],
            ),
        ]

        # Insert barriers before the for loop and at the end of the for loop
        tkw.insert_before(pipeline_loop.KERNEL, tkw.SharedMemoryBarrier())
        tkw.insert_at_end(pipeline_loop.KERNEL, tkw.SharedMemoryBarrier())

        # Apply the cluster-based reordering to the KERNEL stage
        tkw.reorder_graph(pipeline_loop.KERNEL, clusters)

        # Apply staggering waves scheduling to allow two waves to execute clusters in parallel with a stagger offset
        tkw.stagger(pipeline_loop.KERNEL)

    # Define compile options
    M_val, N_val, K_val = shape
    options = WaveCompileOptions(
        subs={
            M: M_val,
            N: N_val,
            K: K_val,
            BLOCK_M: 128,
            BLOCK_N: 256,
            BLOCK_K: 64,
            ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
            ADDRESS_SPACE_0: GLOBAL_ADDRESS_SPACE,
            READ_SHARED_DELAY: 1,
            WRITE_SHARED_DELAY: 1,
            READ_GLOBAL_DELAY: 2,
            WRITE_GLOBAL_DELAY: 2,
            MMA_DELAY: 1,
            VALU_DELAY: 1,
            SHUFFLE_DELAY: 1,
            SHARED_MEMORY_UNITS: 4,
            GLOBAL_MEMORY_UNITS: 4,
            MMA_UNITS: 4,
            VALU_UNITS: 8,
            SHUFFLE_UNITS: 8,
        },
        canonicalize=True,
        schedule=SchedulingType.MANUAL,
        print_ir_after="all" if is_debug else [],
    )

    # Set runtime configuration for execution
    options = set_default_run_config(options)

    # Compile the kernel with the advanced schedule
    gemm_prefetch = wave_compile(options, gemm_prefetch, advanced_schedule)

    # Create test data
    a = torch.randn(shape[0], shape[2], dtype=torch.bfloat16, device="cuda")
    b = torch.randn(shape[1], shape[2], dtype=torch.bfloat16, device="cuda")
    c = torch.zeros(shape[0], shape[1], dtype=torch.float32, device="cuda")

    # Run the kernel
    gemm_prefetch(a, b, c)

    if is_debug:
        print(gemm_prefetch.asm)

    expected = torch.matmul(a, b.t()).to(torch.float32)
    assert torch.allclose(c, expected, rtol=1e-2, atol=1e-2)

    print("GEMM advanced scheduling test passed!")


def test_async_gemm_schedule(is_debug=False):
    """
    GEMM scheduling with async global_to_shared operations and ping-pong buffering.

    This example uses the following scheduling techniques with GatherToLDS:

    1. Async Global-to-Shared: Uses GatherToLDS to combine global load + shared write into a single async operation
    2. Partitioning: Splits MMA operations by K dimension to interleave compute with memory ops
    3. Clustering: Groups instructions to define execution order
    4. Wave Priority: Uses SetWavePrio to adjust compute wave priorities
    5. Stagger: Stagger the waves to allow two waves to execute clusters in parallel with a stagger offset
    6. Barriers: Inserts explicit synchronization and memory counter waits

    The schedule uses async operations to overlap global-to-shared transfers with compute operations
    from different iterations.
    """
    # shape: tuple[int, int, int] = (128, 256, 1024)
    # mfma_variant: tkw.MMAType = tkw.MMAType.F32_16x16x16_F16

    shape: tuple[int, int, int] = (40960, 256, 512)
    mfma_variant: tkw.MMAType = tkw.MMAType.F32_16x16x32_F16

    # Symbol definitions
    M = tkl.sym.M
    N = tkl.sym.N
    K = tkl.sym.K
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N
    BLOCK_K = tkl.sym.BLOCK_K
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
    ADDRESS_SPACE_0 = tkl.sym.ADDRESS_SPACE_0

    # Basic constraints needed for compilation
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.TilingConstraint(K, BLOCK_K)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 4)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 2)]

    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            mma_type=mfma_variant,
        )
    ]

    # Define the kernel (same as before, but we'll apply advanced scheduling)
    @tkw.wave(constraints)
    def gemm_prefetch(
        a: tkl.Memory[M, K, ADDRESS_SPACE, tkl.bf16],
        b: tkl.Memory[N, K, ADDRESS_SPACE, tkl.bf16],
        c: tkl.Memory[M, N, ADDRESS_SPACE_0, tkl.f32],
    ):
        c_reg = tkl.Register[M, N, tkl.f32](0.0)

        @tkw.iterate(K, init_args=[c_reg], tag="k_loop")
        def repeat(
            acc: tkl.Register[M, N, tkl.f32],
        ) -> tkl.Register[M, N, tkl.f32]:
            a_reg = tkw.read(a, tag="read_a")
            b_reg = tkw.read(b, tag="read_b")
            acc = tkw.mma(a_reg, b_reg, acc, tag="mma")
            return acc

        tkw.write(repeat, c)

    # Define the schedule
    @wave_schedule.wave_schedule()
    def async_gemm_schedule():
        """
        Scheduling with cluster-based reordering and ping-pong buffering.

        The schedule creates an instruction ordering that:
        1. Uses async global_to_shared operations (GatherToLDS) to overlap global load + shared write
        2. Interleaves compute (MMA) with memory operations
        3. Uses wave priorities to adjust compute wave priorities
        4. Implements ping-pong buffering via stagger() for double buffering
        5. Places barriers for synchronization
        """
        # Get nodes to be manipulated in the schedule.
        k_loop = tkw.get_node_by_tag("k_loop")

        # Get all nodes with tag "read_a" - includes both Read and GatherToLDS nodes
        all_read_a = tkw.get_node_by_tag("read_a")
        global_to_shared_a = tkw.filter_nodes(all_read_a, node_type=tkw.GatherToLDS)
        shared_load_a = tkw.filter_nodes(all_read_a, node_type=tkw.Read)

        # Get all nodes with tag "read_b" - includes both Read and GatherToLDS nodes
        all_read_b = tkw.get_node_by_tag("read_b")
        global_to_shared_b = tkw.filter_nodes(all_read_b, node_type=tkw.GatherToLDS)
        shared_load_b = tkw.filter_nodes(all_read_b, node_type=tkw.Read)

        mma = tkw.get_node_by_tag("mma")

        pipeline_loop = tkw.pipeline(k_loop)
        # First, create the basic 2-stage pipeline
        with pipeline_loop as pl:
            pl.set_stage(
                [
                    (global_to_shared_a, global_to_shared_b),
                    (),
                ],
            )
            pl.set_stage(
                [
                    (shared_load_a, shared_load_b),
                    (mma,),
                ],
            )

        # Now apply advanced scheduling to the KERNEL stage
        global_to_shared_a = tkw.filter_nodes(
            global_to_shared_a, subgraph=pipeline_loop.KERNEL
        )
        shared_load_a = tkw.filter_nodes(shared_load_a, subgraph=pipeline_loop.KERNEL)
        global_to_shared_b = tkw.filter_nodes(
            global_to_shared_b, subgraph=pipeline_loop.KERNEL
        )
        shared_load_b = tkw.filter_nodes(shared_load_b, subgraph=pipeline_loop.KERNEL)
        mma = tkw.filter_nodes(mma, subgraph=pipeline_loop.KERNEL)

        mma_0, mma_1 = tkw.partition_by_dim(mma, dim=K, num_partitions=2)
        shared_load_a_0, shared_load_a_1 = tkw.partition_by_dim(
            shared_load_a, dim=K, num_partitions=2
        )
        shared_load_b_0, shared_load_b_1 = tkw.partition_by_dim(
            shared_load_b, dim=K, num_partitions=2
        )

        independent_global_count = len(global_to_shared_a) + len(global_to_shared_b)

        clusters = [
            tkw.cluster(
                [
                    shared_load_a_0,
                    shared_load_b_0,
                    tkw.SchedulingBarrier([]),
                    global_to_shared_a,
                    global_to_shared_b,
                    tkw.SchedulingBarrier([]),
                    tkw.WorkgroupBarrier(),
                    tkw.SchedulingBarrier([]),
                ],
            ),
            tkw.cluster(
                [
                    tkw.SetWavePrio(1),
                    mma_0,
                    tkw.SetWavePrio(0),
                    tkw.SchedulingBarrier([]),
                    tkw.MemoryCounterWait(load=independent_global_count),
                    tkw.WorkgroupBarrier(),
                    tkw.SchedulingBarrier([]),
                ],
            ),
            tkw.cluster(
                [
                    shared_load_a_1,
                    shared_load_b_1,
                    tkw.SchedulingBarrier([]),
                    tkw.MemoryCounterWait(load=0),
                    tkw.WorkgroupBarrier(),
                    tkw.SchedulingBarrier([]),
                ],
            ),
            tkw.cluster(
                [
                    tkw.SetWavePrio(1),
                    mma_1,
                    tkw.SetWavePrio(0),
                    tkw.SchedulingBarrier([]),
                ],
            ),
        ]

        # Insert barriers before the for loop and at the end of the for loop
        tkw.insert_before(pipeline_loop.KERNEL, tkw.SharedMemoryBarrier())
        tkw.insert_at_end(pipeline_loop.KERNEL, tkw.SharedMemoryBarrier())

        # Apply the cluster-based reordering to the KERNEL stage
        tkw.reorder_graph(pipeline_loop.KERNEL, clusters)

        # Apply staggering waves scheduling to allow two waves to execute clusters in parallel with a stagger offset
        tkw.stagger(pipeline_loop.KERNEL)

    # Define compile options
    M_val, N_val, K_val = shape
    options = WaveCompileOptions(
        subs={
            M: M_val,
            N: N_val,
            K: K_val,
            BLOCK_M: 128,
            BLOCK_N: 256,
            BLOCK_K: 64,
            ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
            ADDRESS_SPACE_0: GLOBAL_ADDRESS_SPACE,
            READ_SHARED_DELAY: 1,
            WRITE_SHARED_DELAY: 1,
            READ_GLOBAL_DELAY: 2,
            WRITE_GLOBAL_DELAY: 2,
            MMA_DELAY: 1,
            VALU_DELAY: 1,
            SHUFFLE_DELAY: 1,
            SHARED_MEMORY_UNITS: 4,
            GLOBAL_MEMORY_UNITS: 4,
            MMA_UNITS: 4,
            VALU_UNITS: 8,
            SHUFFLE_UNITS: 8,
        },
        canonicalize=True,
        schedule=SchedulingType.MANUAL,
        print_ir_after="all" if is_debug else [],
        use_global_to_shared=True,
        print_mlir=True,
    )

    # Set runtime configuration for execution
    options = set_default_run_config(options)

    # Compile the kernel with the advanced schedule
    gemm_prefetch = wave_compile(options, gemm_prefetch, async_gemm_schedule)

    # Create test data
    a = torch.randn(shape[0], shape[2], dtype=torch.bfloat16, device="cuda")
    b = torch.randn(shape[1], shape[2], dtype=torch.bfloat16, device="cuda")
    c = torch.zeros(shape[0], shape[1], dtype=torch.float32, device="cuda")

    # Run the kernel
    for i in range(100):
        gemm_prefetch(a, b, c)

    if is_debug:
        print(gemm_prefetch.asm)

    expected = torch.matmul(a, b.t()).to(torch.float32)
    assert torch.allclose(c, expected, rtol=1e-2, atol=1e-2)

    print("Async GEMM schedule with global_to_shared test passed!")


def test_async_gemm_schedule_triple_buffering(is_debug=False):
    """
    GEMM scheduling with async global_to_shared operations and triple buffering..

    This example uses the following scheduling techniques with GatherToLDS:

    1. Async Global-to-Shared: Uses GatherToLDS to combine global load + shared write into a single async operation
    2. Partitioning: Splits MMA operations by K dimension to interleave compute with memory ops
    3. Clustering: Groups instructions to define execution order
    4. Wave Priority: Uses SetWavePrio to adjust compute wave priorities
    5. Stagger: Stagger the waves to allow two waves to execute clusters in parallel with a stagger offset
    6. Barriers: Inserts explicit synchronization and memory counter waits

    The schedule uses async operations to overlap global-to-shared transfers with compute operations
    from different iterations. It also uses a 3 stage pipeline to triple buffer the shared memory which allows for 2 memory prefetches before the loop starts.
    """
    # shape: tuple[int, int, int] = (256, 256, 1024)
    shape: tuple[int, int, int] = (40960, 256, 512)
    mfma_variant: tkw.MMAType = tkw.MMAType.F32_16x16x32_F16

    manual_triplebuffer = """
       #map = affine_map<()[s0, s1, s2] -> (s1 * 32 + s2 * 64 + s0 floordiv 8 - ((s1 * 32 + s0 floordiv 8) floordiv 64) * 64)>
        #map1 = affine_map<()[s0] -> ((s0 floordiv 8) mod 8)>
        #map2 = affine_map<()[s0] -> (s0 mod 8)>
        #map3 = affine_map<()[s0] -> (s0 * 8)>
        #map4 = affine_map<()[s0, s1] -> (s1 * 32 + (s0 floordiv 64) * 8 - ((s1 * 4 + s0 floordiv 64) floordiv 8) * 64)>
        #map5 = affine_map<()[s0, s1, s2] -> (s1 * 8 + s2 * 32 + s0 floordiv 32 - ((s1 * 8 + s0 floordiv 32) floordiv 32) * 32)>
        #map6 = affine_map<()[s0] -> (s0 * 2 - (s0 floordiv 32) * 64)>
        #map7 = affine_map<()[s0, s1] -> (s1 * 8 + (s0 floordiv 64) * 2 - ((s1 * 4 + s0 floordiv 64) floordiv 16) * 32)>
        #map8 = affine_map<()[s0, s1, s2] -> (s1 * 8 + s2 * 32 + s0 floordiv 32 - ((s1 * 8 + s0 floordiv 32 + 16) floordiv 32) * 32 + 16)>
        #map9 = affine_map<()[s0, s1] -> (s1 * 8 + (s0 floordiv 64) * 2 - ((s1 * 4 + s0 floordiv 64 + 8) floordiv 16) * 32 + 16)>
        #map10 = affine_map<()[s0, s1] -> (s1 * 4 + s0 floordiv 64)>
        #map11 = affine_map<()[s0] -> (s0 mod 16 + (s0 floordiv 64) * 16)>
        #map12 = affine_map<()[s0] -> ((s0 mod 64) floordiv 16)>
        #map13 = affine_map<()[s0, s1] -> (s0 + s1 * 16 - (s0 floordiv 16) * 16)>
        #map14 = affine_map<()[s0] -> (((s0 mod 64) floordiv 16) * 8)>
        #map15 = affine_map<()[s0] -> ((s0 mod 64) floordiv 16 + 4)>
        #map16 = affine_map<()[s0] -> (((s0 mod 64) floordiv 16) * 8 + 32)>
        #map17 = affine_map<()[s0, s1] -> (s0 * 64 + s1 * 8 + 64)>
        #map18 = affine_map<()[s0, s1] -> (s0 * 64 + s1 * 2 - (s1 floordiv 32) * 64 + 64)>
        #map19 = affine_map<()[s0, s1] -> (s0 * 64 + (s1 floordiv 64) * 16 + ((s1 mod 64) floordiv 16) * 4)>
        #map20 = affine_map<()[s0, s1, s2] -> (s0 + s1 * 32 + s2 * 16 - (s0 floordiv 16) * 16)>
        #map21 = affine_map<()[s0, s1] -> (s0 * 64 + (s1 floordiv 64) * 16 + ((s1 mod 64) floordiv 16) * 4 + 1)>
        #map22 = affine_map<()[s0, s1] -> (s0 * 64 + (s1 floordiv 64) * 16 + ((s1 mod 64) floordiv 16) * 4 + 2)>
        #map23 = affine_map<()[s0, s1] -> (s0 * 64 + (s1 floordiv 64) * 16 + ((s1 mod 64) floordiv 16) * 4 + 3)>
        #translation = #iree_codegen.translation_info<pipeline = None workgroup_size = [256, 2, 1] subgroup_size = 64>
        module attributes {transform.with_named_sequence} {
        stream.executable private @gemm_prefetch {
            stream.executable.export public @gemm_prefetch workgroups() -> (index, index, index) {
            %c4 = arith.constant 4 : index
            %c8 = arith.constant 8 : index
            %c1 = arith.constant 1 : index
            stream.return %c4, %c8, %c1 : index, index, index
            }
            builtin.module {
            func.func @gemm_prefetch(%arg0: !stream.binding, %arg1: !stream.binding, %arg2: !stream.binding) attributes {translation_info = #translation} {
                %c4_i32 = arith.constant 4 : i32
                %c232_i14 = arith.constant 232 : i14
                %c0_i32 = arith.constant 0 : i32
                %c3 = arith.constant 3 : index
                %c1073741823 = arith.constant 1073741823 : index
                %c2147483645_i64 = arith.constant 2147483645 : i64
                %c1073741822 = arith.constant 1073741822 : index
                %c232 = arith.constant 232 : index
                %c1 = arith.constant 1 : index
                %c2 = arith.constant 2 : index
                %c8192 = arith.constant 8192 : index
                %c20480 = arith.constant 20480 : index
                %c16384 = arith.constant 16384 : index
                %c24576 = arith.constant 24576 : index
                %c28672 = arith.constant 28672 : index
                %cst = arith.constant dense<0.000000e+00> : vector<4xf32>
                %c0 = arith.constant 0 : index
                %0 = stream.binding.subspan %arg0[%c0] : !stream.binding -> memref<bf16>
                %1 = stream.binding.subspan %arg1[%c0] : !stream.binding -> memref<bf16>
                %2 = stream.binding.subspan %arg2[%c0] : !stream.binding -> memref<f32>
                %block_id_x = gpu.block_id  x upper_bound 4
                %block_id_y = gpu.block_id  y upper_bound 8
                %thread_id_x = gpu.thread_id  x upper_bound 256
                %thread_id_y = gpu.thread_id  y upper_bound 2
                %reinterpret_cast = memref.reinterpret_cast %0 to offset: [%c0], sizes: [256, 232], strides: [232, 1] : memref<bf16> to memref<256x232xbf16, strided<[232, 1], offset: ?>>
                %reinterpret_cast_0 = memref.reinterpret_cast %1 to offset: [%c0], sizes: [256, 232], strides: [232, 1] : memref<bf16> to memref<256x232xbf16, strided<[232, 1], offset: ?>>
                %reinterpret_cast_1 = memref.reinterpret_cast %2 to offset: [%c0], sizes: [256, 256], strides: [256, 1] : memref<f32> to memref<256x256xf32, strided<[256, 1], offset: ?>>
                %view = memref.alloc() : memref<32x64xbf16, #gpu.address_space<workgroup>>
                %view_2 = memref.alloc() : memref<32x64xbf16, #gpu.address_space<workgroup>>
                %view_3 = memref.alloc() : memref<64x64xbf16, #gpu.address_space<workgroup>>
                %view_4 = memref.alloc() : memref<64x64xbf16, #gpu.address_space<workgroup>>
                %view_5 = memref.alloc() : memref<64x64xbf16, #gpu.address_space<workgroup>>
                %view_6 = memref.alloc() : memref<32x64xbf16, #gpu.address_space<workgroup>>

                %3 = affine.apply #map()[%thread_id_x, %thread_id_y, %block_id_x]
                %4 = affine.apply #map1()[%thread_id_x]
                %5 = affine.apply #map2()[%thread_id_x]
                %6 = arith.xori %5, %4 : index
                %7 = affine.apply #map3()[%6]
                %8 = affine.apply #map4()[%thread_id_x, %thread_id_y]
                %9 = arith.index_cast %8 : index to i32
                %10 = llvm.call_intrinsic "llvm.amdgcn.readfirstlane"(%9) : (i32) -> i32
                %11 = arith.index_cast %10 : i32 to index
                %12 = llvm.call_intrinsic "llvm.amdgcn.readfirstlane"(%c0_i32) : (i32) -> i32
                %13 = arith.index_cast %12 : i32 to index
                %14 = arith.muli %3, %c232 overflow<nsw> : index
                %15 = arith.addi %14, %7 overflow<nsw> : index
                %base_buffer, %offset, %sizes:2, %strides:2 = memref.extract_strided_metadata %reinterpret_cast : memref<256x232xbf16, strided<[232, 1], offset: ?>> -> memref<bf16>, index, index, index, index, index
                %reinterpret_cast_5 = memref.reinterpret_cast %0 to offset: [%offset], sizes: [%c1073741822], strides: [1] : memref<bf16> to memref<?xbf16, strided<[1], offset: ?>>
                %16 = amdgpu.fat_raw_buffer_cast %reinterpret_cast_5 validBytes(%c2147483645_i64) cacheSwizzleStride(%c232_i14) resetOffset : memref<?xbf16, strided<[1], offset: ?>> to memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>
                %17 = arith.cmpi slt, %7, %c232 : index
                %18 = arith.select %17, %15, %c1073741823 : index

                amdgpu.gather_to_lds %16[%18], %view_4[%11, %13] : vector<8xbf16>, memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, memref<64x64xbf16, #gpu.address_space<workgroup>>
                %19 = affine.apply #map5()[%thread_id_x, %thread_id_y, %block_id_y]
                %20 = affine.apply #map6()[%thread_id_x]
                %21 = affine.apply #map7()[%thread_id_x, %thread_id_y]
                %22 = arith.index_cast %21 : index to i32
                %23 = llvm.call_intrinsic "llvm.amdgcn.readfirstlane"(%22) : (i32) -> i32
                %24 = arith.index_cast %23 : i32 to index
                %25 = llvm.call_intrinsic "llvm.amdgcn.readfirstlane"(%c0_i32) : (i32) -> i32
                %26 = arith.index_cast %25 : i32 to index
                %27 = arith.muli %19, %c232 overflow<nsw> : index
                %28 = arith.addi %27, %20 overflow<nsw> : index
                %base_buffer_6, %offset_7, %sizes_8:2, %strides_9:2 = memref.extract_strided_metadata %reinterpret_cast_0 : memref<256x232xbf16, strided<[232, 1], offset: ?>> -> memref<bf16>, index, index, index, index, index
                %reinterpret_cast_10 = memref.reinterpret_cast %1 to offset: [%offset_7], sizes: [%c1073741822], strides: [1] : memref<bf16> to memref<?xbf16, strided<[1], offset: ?>>
                %29 = amdgpu.fat_raw_buffer_cast %reinterpret_cast_10 validBytes(%c2147483645_i64) cacheSwizzleStride(%c232_i14) resetOffset : memref<?xbf16, strided<[1], offset: ?>> to memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>
                %30 = arith.cmpi slt, %20, %c232 : index
                %31 = arith.select %30, %28, %c1073741823 : index
                amdgpu.gather_to_lds %29[%31], %view_2[%24, %26] : vector<2xbf16>, memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, memref<32x64xbf16, #gpu.address_space<workgroup>>
                %32 = affine.apply #map8()[%thread_id_x, %thread_id_y, %block_id_y]
                %33 = affine.apply #map9()[%thread_id_x, %thread_id_y]
                %34 = arith.index_cast %33 : index to i32
                %35 = llvm.call_intrinsic "llvm.amdgcn.readfirstlane"(%34) : (i32) -> i32
                %36 = arith.index_cast %35 : i32 to index
                %37 = llvm.call_intrinsic "llvm.amdgcn.readfirstlane"(%c0_i32) : (i32) -> i32
                %38 = arith.index_cast %37 : i32 to index
                %39 = arith.muli %32, %c232 overflow<nsw> : index
                %40 = arith.addi %39, %20 overflow<nsw> : index
                %41 = arith.select %30, %40, %c1073741823 : index
                amdgpu.gather_to_lds %29[%41], %view_2[%36, %38] : vector<2xbf16>, memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, memref<32x64xbf16, #gpu.address_space<workgroup>>

                %110 = affine.apply #map17()[%c0, %6]
                %111 = arith.addi %14, %110 overflow<nsw> : index
                %112 = arith.cmpi slt, %110, %c232 : index
                %113 = arith.select %112, %111, %c1073741823 : index
                amdgpu.gather_to_lds %16[%113], %view_3[%11, %13] : vector<8xbf16>, memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, memref<64x64xbf16, #gpu.address_space<workgroup>>

                %120 = affine.apply #map18()[%c0, %thread_id_x]
                %122 = arith.addi %27, %120 overflow<nsw> : index
                %121 = arith.cmpi slt, %120, %c232 : index
                %123 = arith.select %121, %122, %c1073741823 : index
                amdgpu.gather_to_lds %29[%123], %view[%24, %26] : vector<2xbf16>, memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, memref<32x64xbf16, #gpu.address_space<workgroup>>

                %125 = arith.addi %39, %120 overflow<nsw> : index
                %126 = arith.select %121, %125, %c1073741823 : index
                amdgpu.gather_to_lds %29[%126], %view[%36, %38] : vector<2xbf16>, memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, memref<32x64xbf16, #gpu.address_space<workgroup>>

                %42 = affine.apply #map10()[%thread_id_x, %thread_id_y]
                %43 = arith.index_cast %42 : index to i32
                %44 = arith.cmpi sge, %43, %c4_i32 : i32
                %45 = arith.cmpi slt, %43, %c4_i32 : i32
                scf.if %44 {
                rocdl.s.barrier
                }
                %46 = llvm.call_intrinsic "llvm.amdgcn.readfirstlane"(%9) : (i32) -> i32
                %47 = arith.index_cast %46 : i32 to index
                %48 = llvm.call_intrinsic "llvm.amdgcn.readfirstlane"(%c0_i32) : (i32) -> i32
                %49 = arith.index_cast %48 : i32 to index
                %50 = llvm.call_intrinsic "llvm.amdgcn.readfirstlane"(%22) : (i32) -> i32
                %51 = arith.index_cast %50 : i32 to index
                %52 = llvm.call_intrinsic "llvm.amdgcn.readfirstlane"(%c0_i32) : (i32) -> i32
                %53 = arith.index_cast %52 : i32 to index
                %54 = llvm.call_intrinsic "llvm.amdgcn.readfirstlane"(%34) : (i32) -> i32
                %55 = arith.index_cast %54 : i32 to index
                %56 = llvm.call_intrinsic "llvm.amdgcn.readfirstlane"(%c0_i32) : (i32) -> i32
                %57 = arith.index_cast %56 : i32 to index
                %58 = affine.apply #map11()[%thread_id_x]
                %59 = affine.apply #map12()[%thread_id_x]
                %60 = arith.xori %59, %5 : index
                %61 = affine.apply #map3()[%60]
                %62 = affine.apply #map13()[%thread_id_x, %thread_id_y]
                %63 = affine.apply #map14()[%thread_id_x]
                %64 = affine.apply #map15()[%thread_id_x]
                %65 = arith.xori %64, %5 : index
                %66 = affine.apply #map3()[%65]
                %67 = affine.apply #map16()[%thread_id_x]
                %68:7 = scf.for %arg3 = %c0 to %c2 step %c1 iter_args(%arg4 = %cst,%a_curr = %view_4, %a_ready = %view_3, %a_fetch = %view_5, %b_curr = %view_2, %b_ready = %view, %b_fetch = %view_6) -> (vector<4xf32>, memref<64x64xbf16, #gpu.address_space<workgroup>>, memref<64x64xbf16, #gpu.address_space<workgroup>>,memref<64x64xbf16, #gpu.address_space<workgroup>>, memref<32x64xbf16, #gpu.address_space<workgroup>>, memref<32x64xbf16, #gpu.address_space<workgroup>>, memref<32x64xbf16, #gpu.address_space<workgroup>>) {
                    //rocdl.s.waitcnt 16371
                    //rocdl.s.waitcnt 883
                    //rocdl.s.waitcnt 8051  //vmcnt 3 and lgmtcnt 0
                    //rocdl.s.waitcnt 112  // vmcnt 0 and lgmcnt 0
                    rocdl.s.waitcnt 8048 // vmcnt 0 ignore others
                    //rocdl.s.waitcnt 16368
                    //amdgpu.lds_barrier
                    rocdl.s.barrier
                    rocdl.s.barrier
                    %arg_30= arith.addi %arg3, %c1 : index
                    %94 = vector.load %a_curr[%58, %61] : memref<64x64xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                    %95 = vector.load %b_curr[%62, %63] : memref<32x64xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                    llvm.call_intrinsic "llvm.amdgcn.sched.barrier"(%c0_i32) : (i32) -> ()
                    %96 = affine.apply #map17()[%arg_30, %6]
                    %97 = arith.addi %14, %96 overflow<nsw> : index
                    %98 = arith.cmpi slt, %96, %c232 : index
                    %99 = arith.select %98, %97, %c1073741823 : index
                    amdgpu.gather_to_lds %16[%99], %a_fetch[%47, %49] : vector<8xbf16>, memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, memref<64x64xbf16, #gpu.address_space<workgroup>>
                    %100 = affine.apply #map18()[%arg_30, %thread_id_x]
                    %101 = arith.addi %27, %100 overflow<nsw> : index
                    %102 = arith.cmpi slt, %100, %c232 : index
                    %103 = arith.select %102, %101, %c1073741823 : index
                    amdgpu.gather_to_lds %29[%103], %b_fetch[%51, %53] : vector<2xbf16>, memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, memref<32x64xbf16, #gpu.address_space<workgroup>>
                    %104 = arith.addi %39, %100 overflow<nsw> : index
                    %105 = arith.select %102, %104, %c1073741823 : index
                    amdgpu.gather_to_lds %29[%105], %b_fetch[%55, %57] : vector<2xbf16>, memref<?xbf16, #amdgpu.address_space<fat_raw_buffer>>, memref<32x64xbf16, #gpu.address_space<workgroup>>
                    llvm.call_intrinsic "llvm.amdgcn.sched.barrier"(%c0_i32) : (i32) -> ()
                    rocdl.s.barrier
                    llvm.call_intrinsic "llvm.amdgcn.sched.barrier"(%c0_i32) : (i32) -> ()
                    rocdl.s.setprio 1
                    %106 = amdgpu.mfma 16x16x32 %94 * %95 + %arg4 blgp = none : vector<8xbf16>, vector<8xbf16>, vector<4xf32>
                    rocdl.s.setprio 0
                    llvm.call_intrinsic "llvm.amdgcn.sched.barrier"(%c0_i32) : (i32) -> ()
                    rocdl.s.barrier
                    llvm.call_intrinsic "llvm.amdgcn.sched.barrier"(%c0_i32) : (i32) -> ()
                    %107 = vector.load %a_curr[%58, %66] : memref<64x64xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                    %108 = vector.load %b_curr[%62, %67] : memref<32x64xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                    llvm.call_intrinsic "llvm.amdgcn.sched.barrier"(%c0_i32) : (i32) -> ()
                    rocdl.s.barrier
                    llvm.call_intrinsic "llvm.amdgcn.sched.barrier"(%c0_i32) : (i32) -> ()
                    rocdl.s.setprio 1
                    %109 = amdgpu.mfma 16x16x32 %107 * %108 + %106 blgp = none : vector<8xbf16>, vector<8xbf16>, vector<4xf32>
                    rocdl.s.setprio 0
                    llvm.call_intrinsic "llvm.amdgcn.sched.barrier"(%c0_i32) : (i32) -> ()
                    scf.yield %109, %a_ready, %a_fetch, %a_curr, %b_ready, %b_fetch, %b_curr : vector<4xf32>, memref<64x64xbf16, #gpu.address_space<workgroup>>, memref<64x64xbf16, #gpu.address_space<workgroup>>,memref<64x64xbf16, #gpu.address_space<workgroup>>, memref<32x64xbf16, #gpu.address_space<workgroup>>, memref<32x64xbf16, #gpu.address_space<workgroup>>,memref<32x64xbf16, #gpu.address_space<workgroup>>
                }
                scf.if %45 {
                rocdl.s.barrier
                }

                %69 = affine.apply #map13()[%thread_id_x, %thread_id_y]
                %70 = affine.apply #map14()[%thread_id_x]
                %71 = vector.load %68#4[%69, %70] : memref<32x64xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %72 = affine.apply #map16()[%thread_id_x]
                %73 = vector.load %68#4[%69, %72] : memref<32x64xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %74 = affine.apply #map11()[%thread_id_x]
                %75 = affine.apply #map12()[%thread_id_x]
                %76 = arith.xori %75, %5 : index
                %77 = affine.apply #map3()[%76]
                %78 = vector.load %68#1[%74, %77] : memref<64x64xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %79 = affine.apply #map15()[%thread_id_x]
                %80 = arith.xori %79, %5 : index
                %81 = affine.apply #map3()[%80]
                %82 = vector.load %68#1[%74, %81] : memref<64x64xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %83 = amdgpu.mfma 16x16x32 %78 * %71 + %68#0 blgp = none : vector<8xbf16>, vector<8xbf16>, vector<4xf32>
                %84 = amdgpu.mfma 16x16x32 %82 * %73 + %83 blgp = none : vector<8xbf16>, vector<8xbf16>, vector<4xf32>

                %94 = vector.load %68#5[%69, %70] : memref<32x64xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %95 = vector.load %68#5[%69, %72] : memref<32x64xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>

                %96 = vector.load %68#2[%74, %77] : memref<64x64xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %97 = vector.load %68#2[%74, %81] : memref<64x64xbf16, #gpu.address_space<workgroup>>, vector<8xbf16>
                %98 = amdgpu.mfma 16x16x32 %96 * %94 + %84 blgp = none : vector<8xbf16>, vector<8xbf16>, vector<4xf32>
                %99 = amdgpu.mfma 16x16x32 %97 * %95 + %98 blgp = none : vector<8xbf16>, vector<8xbf16>, vector<4xf32>

                %85 = vector.extract_strided_slice %99 {offsets = [0], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>

                %86 = affine.apply #map19()[%block_id_x, %thread_id_x]
                %87 = affine.apply #map20()[%thread_id_x, %block_id_y, %thread_id_y]
                vector.store %85, %reinterpret_cast_1[%86, %87] : memref<256x256xf32, strided<[256, 1], offset: ?>>, vector<1xf32>
                %88 = vector.extract_strided_slice %99 {offsets = [1], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %89 = affine.apply #map21()[%block_id_x, %thread_id_x]
                vector.store %88, %reinterpret_cast_1[%89, %87] : memref<256x256xf32, strided<[256, 1], offset: ?>>, vector<1xf32>
                %90 = vector.extract_strided_slice %99 {offsets = [2], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %91 = affine.apply #map22()[%block_id_x, %thread_id_x]
                vector.store %90, %reinterpret_cast_1[%91, %87] : memref<256x256xf32, strided<[256, 1], offset: ?>>, vector<1xf32>
                %92 = vector.extract_strided_slice %99 {offsets = [3], sizes = [1], strides = [1]} : vector<4xf32> to vector<1xf32>
                %93 = affine.apply #map23()[%block_id_x, %thread_id_x]
                vector.store %92, %reinterpret_cast_1[%93, %87] : memref<256x256xf32, strided<[256, 1], offset: ?>>, vector<1xf32>

                return
            }
            }
        }
        func.func @isolated_benchmark$async(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view, %arg2: !hal.buffer_view, %arg3: !hal.fence, %arg4: !hal.fence) -> !hal.buffer_view {
            %0 = hal.tensor.import wait(%arg3) => %arg0 : !hal.buffer_view -> tensor<256x232xbf16>
            %1 = hal.tensor.import wait(%arg3) => %arg1 : !hal.buffer_view -> tensor<256x232xbf16>
            %2 = hal.tensor.import wait(%arg3) => %arg2 : !hal.buffer_view -> tensor<256x256xf32>
            %3 = flow.dispatch @gemm_prefetch::@gemm_prefetch(%0, %1, %2) : (tensor<256x232xbf16>, tensor<256x232xbf16>, tensor<256x256xf32>) -> %2
            %4 = hal.tensor.barrier join(%3 : tensor<256x256xf32>) => %arg4 : !hal.fence
            %5 = hal.tensor.export %4 : tensor<256x256xf32> -> !hal.buffer_view
            return %5 : !hal.buffer_view
        }
        }
    """
    # Symbol definitions
    M = tkl.sym.M
    N = tkl.sym.N
    K = tkl.sym.K
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N
    BLOCK_K = tkl.sym.BLOCK_K
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE
    ADDRESS_SPACE_0 = tkl.sym.ADDRESS_SPACE_0

    # Basic constraints needed for compilation
    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.TilingConstraint(K, BLOCK_K)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / 4)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 2)]

    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=64,
            mma_type=mfma_variant,
        )
    ]

    # Define the kernel (same as before, but we'll apply advanced scheduling)
    @tkw.wave(constraints)
    def gemm_prefetch(
        a: tkl.Memory[M, K, ADDRESS_SPACE, tkl.bf16],
        b: tkl.Memory[N, K, ADDRESS_SPACE, tkl.bf16],
        c: tkl.Memory[M, N, ADDRESS_SPACE_0, tkl.f32],
    ):
        c_reg = tkl.Register[M, N, tkl.f32](0.0)

        @tkw.iterate(K, init_args=[c_reg], tag="k_loop")
        def repeat(
            acc: tkl.Register[M, N, tkl.f32],
        ) -> tkl.Register[M, N, tkl.f32]:
            a_reg = tkw.read(a, tag="read_a")
            b_reg = tkw.read(b, tag="read_b")
            acc = tkw.mma(a_reg, b_reg, acc, tag="mma")
            return acc

        tkw.write(repeat, c)

    # Define the schedule
    @wave_schedule.wave_schedule()
    def async_gemm_schedule_triple_buffering():
        """
        Scheduling with cluster-based reordering and ping-pong buffering.

        The schedule creates an instruction ordering that:
        1. Uses async global_to_shared operations (GatherToLDS) to overlap global load + shared write
        2. Interleaves compute (MMA) with memory operations
        3. Uses wave priorities to adjust compute wave priorities
        4. Implements ping-pong buffering via stagger() for double buffering
        5. Places barriers for synchronization
        """
        # Get nodes to be manipulated in the schedule.
        k_loop = tkw.get_node_by_tag("k_loop")

        # Get all nodes with tag "read_a" - includes both Read and GatherToLDS nodes
        all_read_a = tkw.get_node_by_tag("read_a")
        global_to_shared_a = tkw.filter_nodes(all_read_a, node_type=tkw.GatherToLDS)
        shared_load_a = tkw.filter_nodes(all_read_a, node_type=tkw.Read)

        # Get all nodes with tag "read_b" - includes both Read and GatherToLDS nodes
        all_read_b = tkw.get_node_by_tag("read_b")
        global_to_shared_b = tkw.filter_nodes(all_read_b, node_type=tkw.GatherToLDS)
        shared_load_b = tkw.filter_nodes(all_read_b, node_type=tkw.Read)

        mma = tkw.get_node_by_tag("mma")

        pipeline_loop = tkw.pipeline(k_loop)
        # First, create the basic 3-stage pipeline
        with pipeline_loop as pl:
            pl.set_stage(
                [
                    (global_to_shared_a, global_to_shared_b),
                    (),
                ],
            )
            pl.set_stage(
                [
                    (),
                    (),
                ],
            )
            pl.set_stage(
                [
                    (shared_load_a, shared_load_b),
                    (mma,),
                ],
            )

        # Now apply advanced scheduling to the KERNEL stage
        global_to_shared_a = tkw.filter_nodes(
            global_to_shared_a, subgraph=pipeline_loop.KERNEL
        )
        shared_load_a = tkw.filter_nodes(shared_load_a, subgraph=pipeline_loop.KERNEL)
        global_to_shared_b = tkw.filter_nodes(
            global_to_shared_b, subgraph=pipeline_loop.KERNEL
        )
        shared_load_b = tkw.filter_nodes(shared_load_b, subgraph=pipeline_loop.KERNEL)
        mma = tkw.filter_nodes(mma, subgraph=pipeline_loop.KERNEL)

        mma_0, mma_1 = tkw.partition_by_dim(mma, dim=K, num_partitions=2)
        shared_load_a_0, shared_load_a_1 = tkw.partition_by_dim(
            shared_load_a, dim=K, num_partitions=2
        )
        shared_load_b_0, shared_load_b_1 = tkw.partition_by_dim(
            shared_load_b, dim=K, num_partitions=2
        )
        independent_global_count = len(global_to_shared_a) + len(global_to_shared_b)

        clusters = [
            tkw.cluster(
                [
                    tkw.MemoryCounterWait(load=independent_global_count),
                    tkw.WorkgroupBarrier(),
                    tkw.WorkgroupBarrier(),
                    # tkw.MemoryCounterWait(load=independent_global_count),
                    shared_load_a_0,
                    shared_load_b_0,
                    tkw.SchedulingBarrier([]),
                    global_to_shared_a,
                    global_to_shared_b,
                    tkw.SchedulingBarrier([]),
                    tkw.WorkgroupBarrier(),
                    tkw.SchedulingBarrier([]),
                ],
            ),
            tkw.cluster(
                [
                    tkw.SetWavePrio(1),
                    mma_0,
                    tkw.SetWavePrio(0),
                    tkw.SchedulingBarrier([]),
                    tkw.WorkgroupBarrier(),
                    tkw.SchedulingBarrier([]),
                ],
            ),
            tkw.cluster(
                [
                    tkw.MemoryCounterWait(load=independent_global_count),
                    shared_load_a_1,
                    shared_load_b_1,
                    tkw.SchedulingBarrier([]),
                    tkw.WorkgroupBarrier(),
                    tkw.SchedulingBarrier([]),
                ],
            ),
            tkw.cluster(
                [
                    tkw.SetWavePrio(1),
                    mma_1,
                    tkw.SetWavePrio(0),
                    tkw.SchedulingBarrier([]),
                    # tkw.WorkgroupBarrier(),
                ],
            ),
        ]

        # Apply the cluster-based reordering to the KERNEL stage
        tkw.reorder_graph(pipeline_loop.KERNEL, clusters)

        # Apply staggering waves scheduling to allow two waves to execute clusters in parallel with a stagger offset
        tkw.stagger(pipeline_loop.KERNEL)

    # Define compile options
    M_val, N_val, K_val = shape
    options = WaveCompileOptions(
        subs={
            M: M_val,
            N: N_val,
            K: K_val,
            BLOCK_M: 128,
            BLOCK_N: 256,
            BLOCK_K: 64,
            ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
            ADDRESS_SPACE_0: GLOBAL_ADDRESS_SPACE,
        },
        canonicalize=True,
        schedule=SchedulingType.MANUAL,
        print_ir_after="all" if is_debug else [],
        use_global_to_shared=True,
        minimize_shared_allocs=False,  ### Important: if you enable the pass , memref.views will be created instead of memref.alloc hence aliasing infromation will be lost and backend will not insert automatic barriers between dependent memory operations. You will need to add barriers by yourself in the scheudle
        print_mlir=True,
        specialize=True,
        # override_mlir=manual_triplebuffer,
    )

    # Set runtime configuration for execution
    options = set_default_run_config(options)

    # Compile the kernel with the advanced schedule
    gemm_prefetch = wave_compile(
        options, gemm_prefetch, async_gemm_schedule_triple_buffering
    )

    # Create test data
    a = torch.randn(shape[0], shape[2], dtype=torch.bfloat16, device="cuda")
    b = torch.randn(shape[1], shape[2], dtype=torch.bfloat16, device="cuda")
    c = torch.zeros(shape[0], shape[1], dtype=torch.float32, device="cuda")

    # Run the kernel
    for i in range(100):
        gemm_prefetch(a, b, c)

    if is_debug:
        print(gemm_prefetch.asm)

    expected = torch.matmul(a, b.t()).to(torch.float32)
    assert torch.allclose(c, expected, rtol=1e-2, atol=1e-2)

    print(
        "Async GEMM schedule using triple buffering with global_to_shared test passed!"
    )


if __name__ == "__main__":
    args = parse_args()

    if args.list_tests:
        list_tests(globals())
        exit(0)

    if not args.test:
        print("Error: --test argument is required")
        print("Use --list_tests to see available tests")
        exit(1)

    success = run_test(args.test, globals(), args.debug, args.repeat)
    exit(0 if success else 1)
