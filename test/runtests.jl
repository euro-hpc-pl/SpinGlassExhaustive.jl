using SpinGlassEngine
using Logging
using SpinGlassNetworks, SpinGlassTensors
using LightGraphs
using LinearAlgebra
using SpinGlassNetworks
using Test

my_tests = []

push!(my_tests,
    "brute_force.jl"
)

for my_test in my_tests
    include(my_test)
end
