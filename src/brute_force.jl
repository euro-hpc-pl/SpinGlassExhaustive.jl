export brute_force_gpu

function kernel(J, energies, σ)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    L = size(σ, 1)

    for i=1:L if tstbit(idx, i) @inbounds σ[i, idx] = 1 end end

    for k=1:L
        @inbounds energies[idx] += J[k, k] * σ[k, idx]
        for l=(k+1):L @inbounds energies[idx] += σ[k, idx] * J[k, l] * σ[l, idx] end
    end
    return
end

function brute_force_gpu(ig::IsingGraph, max_states::Int=100)
    L = nv(ig)
    N = 2^L

    energies = CUDA.zeros(N)
    σ = CUDA.zeros(Int, L, N) .- 1
    J = CUDA.CuArray(couplings(ig) + Diagonal(biases(ig)))

    k = 10
    @cuda threads=2^k blocks=2^(L-k) kernel(J, energies, σ)

    energies_cpu = Array(energies)
    σ_cpu = Array(σ)

    perm = partialsortperm(energies_cpu, 1:max_states)
    Spectrum(energies_cpu[perm], σ_cpu[:, perm])
end
