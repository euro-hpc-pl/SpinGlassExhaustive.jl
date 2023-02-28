using SpinGlassExhaustive

my_tests = ["ising.jl", "utils.jl", "brute_force.jl"]

for my_test ∈ my_tests
    include(my_test)
end