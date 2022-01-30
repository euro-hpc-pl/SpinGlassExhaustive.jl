
# function ising(cubo)
#     N = size(qubo)[1]

#     low_en = CUDA.zeros(2^N)
#     low_st = CUDA.zeros(2*N)

#     @inline function _energy(q, graph)
#         F = 0
#         N = size(graph)[1]

#         for i in 1:N
#             F -= graph[i,i]*q[i]
#             for j in 1:N
#                 low, high = i < j ? (i, j) : (j, i)
#                 F -= graph[low,high]*q[i]*q[j]
#             end
#         end
#         return F
#     end

#     @inline function _dec_to_binary(state_code)

#         binaryNum = @SVector [0 for i in 1:N+1]

#         s = 1
#         while state_code > 0
#             @set! binaryNum[s] = state_code%2
#             state_code=div(state_code,2)
#             s+=1
#         end

#         return binaryNum
#     end

#     @inline function kernel(qubo, low_en, low_st)
#         N = size(qubo)[1]

#         i = blockIdx().x
#         j = threadIdx().x

#         state_code = (i - 1) * blockDim().x + j

#         q = _decToBinary(state_code)
#         F = _energy(q, qubo)

#         low_en[(i - 1) * blockDim().x + j] = F

#         return
#     end

#     k = 2
#     @cuda blocks=(2^(N-k)) threads=(2^k) kernel(qubo, low_en, low_st)

#     return low_en
# end