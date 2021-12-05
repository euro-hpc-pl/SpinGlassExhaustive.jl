"""
Main module for `SpinGlassExhaustive.jl` -- a Julia package for brute-force spin-glass problems with CUDA.
"""

module SpinGlassExhaustive
using CUDA
using Bits
using BenchmarkTools
using StaticArrays
using Setfield
using DocStringExtensions
using SpinGlassNetworks
eval(Expr(:export, names(SpinGlassExhaustive)...))

include("bitonicsort.jl")
include("ising.jl")
include("utils.jl")
include("brute_force.jl")

end
