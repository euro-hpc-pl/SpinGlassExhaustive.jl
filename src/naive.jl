export brute_force

function naive_energy_kernel(J, energies, σ)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x

    L = size(σ, 1)
    for j ∈ idx:stride:length(energies)
        for i=1:L if tstbit(j-1, i) @inbounds σ[i, j] = 1 end end
        en = 0
        for k ∈ 1:L
            @inbounds en += J[k, k] * σ[k, j]
            for l ∈ k+1:L @inbounds en += σ[k, j] * J[k, l] * σ[l, j] end
        end
        energies[j] = en
    end
    return
end

function SpinGlassNetworks.brute_force(
    ig::IsingGraph,
    ::Val{:GPU};
    num_states::Int=1
)
    L = nv(ig)
    N = 2^L
    σ = CUDA.fill(Int32(-1), L, N)
    JJ = couplings(ig)
    J = CUDA.CuArray(JJ + Diagonal(biases(ig)))
    energies = CUDA.zeros(eltype(JJ), N)

    th = 2 ^ 10 # this should eventually vary
    bl = cld(N, th)

    @cuda threads=th blocks=bl naive_energy_kernel(J, energies, σ)

    perm = sortperm(energies)[1:num_states]
    energies_cpu = Array(view(energies, perm))
    σ_cpu = Array(view(σ, :, perm))
    Spectrum(energies_cpu, [σ_cpu[:, i] for i ∈ 1:size(σ_cpu, 2)])
end
