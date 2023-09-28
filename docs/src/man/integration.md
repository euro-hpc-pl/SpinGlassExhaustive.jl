```@setup SpinGlassExhaustive
using SpinGlassExhaustive

import Pkg;
Pkg.add("CUDA")
using CUDA

Pkg.add("SpinGlassEngine")
using SpinGlassEngine
```
# Integration SpinGlassExhaustive with other EuroHPC packages 
As part of the Euro-HPC project, a number of tools were developed to solve the Ising problem. In this section, we will present how to benchmark algorithms from SpinGlassExhaustive, SpinGlassEngine.jl and SpinGlassNetworks.jl.


```@repl SpinGlassExhaustive
instance = "benchmarks/pathological/test_3_4_3.txt"

bench(instance)
```