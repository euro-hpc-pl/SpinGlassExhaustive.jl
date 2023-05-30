using Documenter, SpinGlassExhaustive

using Pkg
Pkg.activate("..")
push!(LOAD_PATH,"../src/")



_pages = [
    "Introduction" => "index.md",
    # "User Guide" => "guide.md",   
    "API Reference" => "api.md",
    # "Library" => "lib/SpinGlassExhaustive.md"
]
# ============================

makedocs(
    # root = "../",
    # source = "src",
    modules = Module[SpinGlassExhaustive],
    sitename="SpinGlassExhaustive.jl",
    authors = "Dariusz Kurzyk, Łukasz Pawela",
    pages = _pages,
    format = Documenter.HTML(prettyurls = false),
    expandfirst = []
    )

deploydocs(
    deps = Deps.pip("pygments", "mkdocs", "python-markdown-math"),
    target = "build",
    repo = "github.com/euro-hpc-pl/SpinGlassExhaustive.jl.git"
)