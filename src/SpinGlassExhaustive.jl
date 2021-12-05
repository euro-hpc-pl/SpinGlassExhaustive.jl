"""
Main module for `SpinGlassExhaustive.jl` -- a Julia package for brute-force spin-glass problems with CUDA.
"""

module SpinGlassExhaustive
using CUDA
using BenchmarkTools
using StaticArrays
using Setfield
using DocStringExtensions
using SpinGlassExhaustive
eval(Expr(:export, names(SpinGlassExhaustive)...))

include("bitonicsort.jl")
include("ising.jl")
include("utils.jl")

end
