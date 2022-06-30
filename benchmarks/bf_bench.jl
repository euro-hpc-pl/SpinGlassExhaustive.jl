using SpinGlassNetworks, LightGraphs, SpinGlassExhaustive
using CUDA, LinearAlgebra
using Bits

function bench_gpu(instance::String)
    ig = SpinGlassEngine.ising_graph(instance)
    graph = couplings(ig) + SpinGlassEngine.Diagonal(biases(ig))
    cu_graph = graph |> cu 
    qubo = graph_to_qubo(graph)
    cu_qubo = qubo |> cu 

    println("Graph size ", size(cu_graph,1))
    
    # Setting SpinGlassEnginge
    nrows = 4
    ncols = 4

    fg = factor_graph(
            ig, cluster_assignment_rule=super_square_lattice((nrows, ncols, 14))
        )
    
    β = 1

    network = PEPSNetwork{true}(
        nrows,
        ncols,
        fg,
        rotation(180),
        β=β,
        bond_dim=24
    )

    println("Result for SpinGlassEngine (low_energy_spectrum)")
    @time spe = low_energy_spectrum(network, 24)
    
    # SpinGlassExhaustive
    N = size(cu_graph,1)
    k = 2
    threadsPerBlock::Int64 = 2^k
    blocksPerGrid::Int64 = 2^(N-k)
    
    # Ising

    energies = CUDA.zeros(2^N)

    println("Result for CUDA SpinGlassExuastive (basic approach)")
    @time @cuda blocks=(blocksPerGrid) threads=(threadsPerBlock) kernel(cu_graph, energies)

    states = sortperm(energies)
    energies = energies[states]
    
    bf1 = energies[1]
    
    # Ising partial
    
    energies = CUDA.zeros(2^N)
    part_st = CUDA.zeros(2^(N-k))
    part_lst = CUDA.zeros(2^(N-k))

    println("Result for CUDA SpinGlassExuastive (partial result)")
    @time @cuda blocks=(blocksPerGrid) threads=(threadsPerBlock) kernel_part(cu_graph, energies, part_lst, part_st)

    idx = sortperm(part_lst)
    states = part_st[idx]
    bf2 = part_lst[idx]
    
    # QUBO
    energies = CUDA.zeros(2^N)

    println("Result for CUDA SpinGlassExuastive (QUBO problem)")
    @time @cuda blocks=(blocksPerGrid) threads=(threadsPerBlock) kernel_qubo(cu_qubo, energies)

    states = sortperm(energies)
    
    offset = get_energy_offset(Array(cu_graph))
    bf3 = Array(sort!(energies[states])).-offset
    
    
    spe.energies[1], bf1[1], bf2[1], bf3[1]
end 

bech_ret = bench_gpu("$(@__DIR__)/benchmarks/pathological/chim_3_4_3.txt")


# sp_cpu = bench_cpu("$(@__DIR__)/pegasus_droplets/2_2_3_00.txt")
# println("*** GPU ***")
# sp_gpu = bench_gpu("$(@__DIR__)/pegasus_droplets/2_2_3_00.txt")
# sp_gpu = bench_gpu("$(@__DIR__)/pegasus_droplets/2_2_3_00.txt")

# @assert sp_gpu.energies ≈ sp_cpu.energies
# @assert all(
#     sp_gpu.states[i] == sp_cpu.states[i]
#     for i ∈ 1:size(sp_cpu.states, 1)
# )
