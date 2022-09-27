using Distributions

function generate_random_graph(d::Int)
    graph = rand(Uniform(-5,5),d, d)
    graph = graph * graph'

    graph
end

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