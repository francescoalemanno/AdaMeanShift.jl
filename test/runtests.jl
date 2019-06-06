using AdaMeanShift
using StaticArrays
using Test

@testset "AdaMeanShift.jl" begin
    vx=-20:0.1:20
    vy=0:0.1:30
    M=[exp(-(x-7)^2/(2*3)-(y-12)^2/(2*2)) for x in vx, y in vy]
    posmax=Tuple(findmax(M)[2])
    posmax.*1.0
    P,h,w,δ = atomic_meanshift(M, @SVector([100.0,100.0]) , @SVector([50.0,50.0]), 0.0,0.0) 
    for i in 1:500
        P,h,w,δ = atomic_meanshift(M, P , h, 0.0,0.0)
    end
    v=Tuple(P).-posmax
    @test sum(abs.(v))<1e-10
end
