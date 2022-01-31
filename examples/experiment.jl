using SpinGlassExhaustive

function exp_qubo()
  N = 8
  graph = generate_random_graph(N)
  qubo = graph_to_qubo(graph)

  cu_qubo = qubo |> cu 

  k = 2

  energies = CUDA.zeros(2^N)

  threadsPerBlock::Int64 = 2^k
  blocksPerGrid::Int64 = 2^(N-k)

  @cuda blocks=(blocksPerGrid) threads=(threadsPerBlock) kernel_qubo(cu_qubo, energies)

  sort!(energies)
 
end

function exp_ising()
  N = 8
  graph = generate_random_graph(N)
  cu_graph = graph |> cu 
  
  k = 2

  energies = CUDA.zeros(2^N)

  threadsPerBlock::Int64 = 2^k
  blocksPerGrid::Int64 = 2^(N-k)

  @cuda blocks=(blocksPerGrid) threads=(threadsPerBlock) kernel_qubo(cu_graph, energies)

  sort!(energies)
 
end
