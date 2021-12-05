
sp_cpu = bench_cpu("$(@__DIR__)/pegasus_droplets/2_2_3_00.txt");
sp_gpu = bench_gpu("$(@__DIR__)/pegasus_droplets/2_2_3_00.txt");

sp_cpu.energies ≈ sp_gpu.energies

println(minimum(sp_gpu.energies))
println(minimum(sp_cpu.energies))

sp_gpu.states[:, begin] == sp_cpu.states[begin]
