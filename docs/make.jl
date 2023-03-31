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

#####
# using Documenter, Example

# makedocs(sitename="My Documentation")

#####


# using Documenter, SpinGlassExhaustive

# format = Documenter.HTML(edit_link = "master",
#                          prettyurls = get(ENV, "CI", nothing) == "true",
# )

# makedocs(
#     clean = true,
#     format = format,
#     sitename = "SpinGlassExhaustive.jl",
#     authors = "Dariusz Kurzyk, Łukasz Pawela",
#     # assets = ["assets/favicon.ico"],
#     pages = [
#         "Home" => "index.md",
#         # "Index" => [
#         #     "lib/content/ising.md"
#         # ]
#         # "Manual" => Any[
#         #     "man/quickstart.md",
#         #     "man/vectors.md",
#         #     "man/states.md",
#         #     "man/functionals.md",
#         #     "man/measurement.md",
#         #     "man/random.md"
#         # ],
#         # "Library" => "lib/SpinGlassExhaustive.md",
#         # Any[
#         #     "lib/content/ising.md"
#         # ]
#     ]
# )

# deploydocs(
#     deps = Deps.pip("pygments", "mkdocs", "python-markdown-math"),
#     target = "build",
#     repo = "github.com/euro-hpc-pl/SpinGlassExhaustive.jl.git"
# )