# using SpinGlassNetworks, LightGraphs, SpinGlassExhaustive
# using CUDA, LinearAlgebra
# using Bits

# function bench_cpu(instance::String, max_states::Int=100)
#     m = 2
#     n = 2
#     t = 24

#     ig = ising_graph(instance)
#     cl = split_into_clusters(ig, super_square_lattice((m, n, t)))
#     @time sp = brute_force(cl[1, 1], num_states=max_states)
#     sp
# end



# function bench_gpu(instance::String, max_states::Int=100)
#     m = 2
#     n = 2
#     t = 24

#     ig = ising_graph(instance)
#     cl = split_into_clusters(ig, super_square_lattice((m, n, t)))
#     @time sp = brute_force(cl[1, 1], :GPU; num_states=max_states)
#     sp
# end

# println("*** CPU ***")
# sp_cpu = bench_cpu("$(@__DIR__)/pegasus_droplets/2_2_3_00.txt")
# sp_cpu = bench_cpu("$(@__DIR__)/pegasus_droplets/2_2_3_00.txt")
# println("*** GPU ***")
# sp_gpu = bench_gpu("$(@__DIR__)/pegasus_droplets/2_2_3_00.txt")
# sp_gpu = bench_gpu("$(@__DIR__)/pegasus_droplets/2_2_3_00.txt")

# @assert sp_gpu.energies ≈ sp_cpu.energies
# @assert all(
#     sp_gpu.states[i] == sp_cpu.states[i]
#     for i ∈ 1:size(sp_cpu.states, 1)
# )
