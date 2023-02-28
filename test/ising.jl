@testset "Compare ising kernel with naive approach" begin
    N = 8
    graph = generate_random_graph(N)
    cu_graph = graph |> cu 
    
    ig = SpinGlassEngine.ising_graph(graph_to_dict(cu_graph))    
    
    res_naive = SpinGlassNetworks.brute_force(ig)
    res_ising_bucket = exhaustive_search(ig)
    
    @test res_ising_bucket.energies[1] ≈ res_ising_bucket.energies[1]

end 

@testset "Compare ising kernel returning partial result with naive approach" begin
    N = 8
    graph = generate_random_graph(N)
    cu_graph = graph |> cu 
    
    ig = SpinGlassEngine.ising_graph(graph_to_dict(cu_graph))    
    
    res_naive = SpinGlassNetworks.brute_force(ig)
    res_ising_bucket = partial_exhaustive_search(ig)
    
    @test res_ising_bucket.energies[1] ≈ res_ising_bucket.energies[1]

end 

@testset "Check conversion of qubo solution to ising model" begin
    N = 8
    graph = generate_random_graph(N)
    cu_graph = graph |> cu 
    
    qubo = graph_to_qubo(graph)

    cu_qubo = qubo |> cu 

    k = 2
  
    energies = CUDA.zeros(2^N)
    qubo_energies = CUDA.zeros(2^N)
  
    threadsPerBlock::Int64 = 2^k
    blocksPerGrid::Int64 = 2^(N-k)
  
    @cuda blocks=(blocksPerGrid) threads=(threadsPerBlock) kernel(cu_graph, energies)
  
    cuda_min_energy = sort!(energies)[1]

    @cuda blocks=(blocksPerGrid) threads=(threadsPerBlock) kernel_qubo(cu_qubo, qubo_energies) 

    offset = get_energy_offset(Array(cu_graph))

    @test cuda_min_energy[1] ≈ sort!(qubo_energies)[1]-offset

end 

@testset "Compare ising kernel with bucket sort returning result with naive approach" begin
    N = 8
    graph = generate_random_graph(N)
    cu_graph = graph |> cu 
    
    ig = SpinGlassEngine.ising_graph(graph_to_dict(cu_graph))    
    
    res_naive = SpinGlassNetworks.brute_force(ig)
    res_ising_bucket = exhaustive_search_bucket(ig)
    
    @test res_ising_bucket.energies[1] ≈ res_ising_bucket.energies[1]

end 