using Distributions
export generate_random_graph
"""
$(SIGNATURES)
- `d::Int`: size of random graph.
Returns random array of size d.
"""
function generate_random_graph(d::Int)
    graph = rand(Uniform(-5,5),d, d)
    graph = graph * graph'

    graph
end

"""
$(SIGNATURES)
- `graph`: graph of the ising model.
Converts Ising model graph to QUBO representation.
"""
function graph_to_qubo(graph)
    N = size(graph,1)
    
    qubo = zeros(N,N)
    for i in 1:N
        qubo[i,i] = 2 * graph[i,i]

        for j in 1:N
            if i!=j
                low, high = sort!([i, j])
                qubo[i,i] -= 2*graph[low,high]
                qubo[low,high] += graph[low,high]*2
            end
        end
    end
    qubo
end

"""
$(SIGNATURES)
- `graph`: graph of the ising model.
Converts Ising model graph to Dict.
"""
function graph_to_dict(graph)
    N = size(graph,1)
    graph_dict = Dict()
    for i in 1:N
        graph_dict[(i, i)] = graph[i,i] #rand(Uniform(-5,5))
        for j in i:N
            graph_dict[(i, j)] = graph[i,j] #rand(Uniform(-5,5))
        end
    end
    graph_dict
end

"""
$(SIGNATURES)
- `graph`: graph of the ising model.
Returns offest between Ising model graph and its QUBO representation.
"""
function get_energy_offset(graph)
    N = size(graph,1)
    offset = 0

    for i in 1:N
        offset += graph[i,i]
        for j in i:N
            if i!=j
                offset -= graph[i,j]
            end
        end
    end
    offset
end
