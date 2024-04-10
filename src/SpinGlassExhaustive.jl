"""
Main module for `SpinGlassExhaustive.jl` -- a Julia package for brute-force spin-glass problems with CUDA.
"""

module SpinGlassExhaustive
    using SpinGlassNetworks
    using Graphs
    using LabelledGraphs
    using Bits
    using LinearAlgebra

    include("naive.jl")
end