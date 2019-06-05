using Documenter, AdaMeanShift

makedocs(;
    modules=[AdaMeanShift],
    format=Documenter.HTML(),
    pages=[
        "Home" => "index.md",
    ],
    repo="https://github.com/francescoalemanno/AdaMeanShift.jl/blob/{commit}{path}#L{line}",
    sitename="AdaMeanShift.jl",
    authors="Francesco Alemanno",
    assets=String[],
)

deploydocs(;
    repo="github.com/francescoalemanno/AdaMeanShift.jl",
)
