function energy_qubo(state_code, graph)
    F = 0
    N = size(graph,1)
    
    for i in 1:N
        
        tstbit(state_code, i) ? qi=1 : continue
        
        F += graph[i,i]*qi  
               
        for j in i+1:N
                       
            tstbit(state_code, j) ? qj=1 : continue
                                   
            F += graph[i,j]*qi*qj
        end
    end
    return F
end

function energy(state_code, graph)
    F = 0
    N = size(graph,1)
    
    for i in 1:N
        
        tstbit(state_code, i) ? qi=1 : qi=-1
                             
        F += graph[i,i]*qi  
               
        for j in i+1:N
                       
            tstbit(state_code, j) ? qj=1 : qj=-1
                                   
            F += graph[i,j]*qi*qj
        end
    end
    return F
end

function kernel(graph, energies)
    N = size(graph,1)

    state_code = (blockIdx().x - 1) * blockDim().x + threadIdx().x 

    F = energy(state_code, graph) |> Float32
    
    @inbounds energies[state_code] = F
          
    return
end


function kernel_qubo(graph, energies)
    N = size(graph,1)

    state_code = (blockIdx().x - 1) * blockDim().x + threadIdx().x 

    F = energy_qubo(state_code, graph) |> Float32
    
    @inbounds energies[state_code] = F
          
    return
end

function kernel_part(graph, energies, part_lst, part_st)
    N = size(graph,1)

    i = blockIdx().x
    j = threadIdx().x

    state_code = (i - 1) * blockDim().x + j 

    F = energy(state_code, graph) |> Float32
    
    @inbounds energies[state_code] = F

    sync_threads()
    
    if j == 1
        k = (i - 1) * blockDim().x + 1
        n = blockDim().x
        
        value=energies[k]
        st=k
        for ii in k:k+n-1
            if value > energies[ii]
                value = energies[ii]
                st=k
            end
        end 
               
        sync_threads()
        
        part_lst[i] = value # low_en[k]
        part_st[i] = st
    end
    
    return
end