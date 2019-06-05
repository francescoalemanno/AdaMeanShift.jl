using Documenter, AdaMeanShift

makedocs(;
    modules=[AdaMeanShift],
    format=Documenter.HTML(),
    pages=[
        "Home" => "index.md",
    ],
    sitename="AdaMeanShift.jl",
    authors="Francesco Alemanno",
    assets=String[],
)

deploydocs(;
    repo="github.com/francescoalemanno/AdaMeanShift.jl.git",
)
