using SpinGlassNetworks
using SpinGlassExhaustive

function bench_cpu(instance::String, max_states::Int=100)
    m = 2
    n = 2
    t = 24

    ig = ising_graph(instance)
    cl = split_into_clusters(ig, super_square_lattice((m, n, t)))
    @time sp = brute_force(cl[1, 1], num_states=max_states)
    sp
end


function bench_gpu(instance::String, max_states::Int=100)
    m = 2
    n = 2
    t = 24

    ig = ising_graph(instance)
    cl = split_into_clusters(ig, super_square_lattice((m, n, t)))
    @time sp = brute_force_gpu(cl[1, 1], num_states=max_states)
    sp
end

sp_cpu = bench_cpu("$(@__DIR__)/pegasus_droplets/2_2_3_00.txt");
sp_gpu = bench_gpu("$(@__DIR__)/pegasus_droplets/2_2_3_00.txt");

sp_cpu.energies ≈ sp_gpu.energies

println(minimum(sp_gpu.energies))
println(minimum(sp_cpu.energies))

@assert sp_gpu.states[:, begin] == sp_cpu.states[begin]
