N = 16

function energy(graph, state_code)
    F = 0
    N = size(graph)[1]
    q = digits(state_code, base=2, pad=N) |> reverse

    for i in 1:N
        F -= graph[i,i]*q[i]  
        for j in 1:N
            low, high = sort!([i, j])
            F -= graph[low,high]*q[i]*q[j]
        end
    end
    return F
end

function dec_to_binary(state_code)
    binaryNum = @SVector [0 for i in 1:N+1]

    s = 1
    while state_code > 0
        @set! binaryNum[s] = state_code%2
        state_code=div(state_code,2)
        s+=1
    end
    
    return binaryNum
end

function generate_random_graph(d::Int)
    graph = rand(Uniform(-5,5),d, d)
    graph = graph * graph'

    graph_dict = Dict()
    for i in 1:d
        graph_dict[(i, i)] = graph[i,i]
        for j in i:d
            graph_dict[(i, j)] = graph[i,j]
        end
    end
end