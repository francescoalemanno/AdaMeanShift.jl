using Documenter, AdaMeanShift

makedocs(;
    modules=[AdaMeanShift],
    format = Documenter.HTML(assets=["assets/invenia.css"]),
    pages=[
        "API" => "index.md",
    ],
    sitename="AdaMeanShift.jl",
    authors="Francesco Alemanno",
)

deploydocs(;
    repo="github.com/francescoalemanno/AdaMeanShift.jl.git",
    target="build"
)
