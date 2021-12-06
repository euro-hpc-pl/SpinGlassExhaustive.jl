export brute_force

function naive_energy_kernel(J, energies, σ)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    L = size(σ, 1)

    for i=1:L if tstbit(idx, i) @inbounds σ[i, idx] = 1 end end

    en = 0.0
    for k=1:L
        en += J[k, k] * σ[k, idx]
        for l=k+1:L en += σ[k, idx] * J[k, l] * σ[l, idx] end
    end
    energies[idx] = en
    return
end

function SpinGlassNetworks.brute_force(ig::IsingGraph, ::Val{:GPU}; num_states::Int=1)
    L = nv(ig)
    N = 2^L
    k = 10 #UGLY HACK!
    energies = CUDA.zeros(N)
    σ = CUDA.fill(Int32(-1), L, N)
    J = couplings(ig) + Diagonal(biases(ig))
    J_dev = CUDA.CuArray(J)
    @cuda threads=2^k blocks=(2^(L-k)) naive_energy_kernel(J_dev, energies, σ)
    perm = sortperm(energies)[1:num_states]
    energies_cpu = Array(view(energies, perm))
    σ_cpu = Array(view(σ, :, perm))
    Spectrum(energies_cpu, [σ_cpu[:, i] for i ∈ 1:size(σ_cpu, 2)])
end
