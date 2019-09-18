using AdaMeanShift
using StaticArrays
using LinearAlgebra
using Test

@testset "AdaMeanShift.jl" begin
    vx=-20:0.1:20
    vy=0:0.1:30
    M=[exp(-(x-7)^2/(2*3)-(y-12)^2/(2*2)) for x in vx, y in vy]
    posmax=Tuple(findmax(M)[2])
    P,h,w,δ = atomic_meanshift(M, @SVector([100.0,100.0]) , @SVector([50.0,50.0]), 0.0,0.0)
    for i in 1:500
        P,h,w,δ = atomic_meanshift(M, P , h, 0.0,0.0)
    end
    v=Tuple(P).-posmax
    @test sum(abs.(v))<1e-7
    @test round(Int,atomic_intensity(M, P, h).intensity) == 1263.0

    Pv=[@SVector([100.0,100.0]) for rep=1:200]
    hv=[@SVector([50.0,50.0]) for rep=1:200]
    wv=[0.0 for rep=1:200]
    XX=meanshift!(M, Pv, hv, wv,Inf,smoothing=0.0,isotropy=0.0)
    @test atomic_meanshift(zeros(0,0),[0.0,0.0],[0.0,0.0],0.0,0.0).modeval===NaN
    @test norm(Tuple(Pv[1]).-posmax)<1e-7
    XX=meanshift!(M, Pv, hv, wv,20.0,smoothing=0.0)
    @test norm(hv[1])≈20.0
    @test norm(Tuple(Pv[1]).-posmax)<1e-7
end


@testset "Helper Methods" begin
    vx=-20:0.1:20
    vy=0:0.1:30
    M=[exp(-(x-7)^2/(2*3)-(y-12)^2/(2*2)) for x in vx, y in vy]
    posmax=Tuple(findmax(M)[2])
    P,h,w,δ = atomic_meanshift(M, [100.0,100.0] , [50.0,50.0], 0.0,0.0)
    for i in 1:500
        P,h,w,δ = atomic_meanshift(M, P , h, 0.0,0.0)
    end
    v=Tuple(P).-posmax
    Pjl=[x for x in Tuple(P)]
    hjl=[x for x in Tuple(h)]
    @test sum(abs.(v))<1e-7
    @test round(Int,atomic_intensity(M, Pjl, hjl).intensity) == 1263.0
end


@testset "Non adaptive routines" begin
    vx=-20:0.1:20
    vy=0:0.1:30
    M=[exp(-(x-7)^2/(2*3)-(y-12)^2/(2*2)) for x in vx, y in vy]
    posmax=Tuple(findmax(M)[2])

    Pv=[@SVector([100.0,100.0])]
    hv=[@SVector([50.0,50.0])]
    wv=[0.0]
    XX=meanshift_nonadaptive!(M, Pv, hv, wv,smoothing=0.0)

    @test norm(Tuple(Pv[1]).-posmax)<1e-7
end


@testset "Ratio Calculation" begin
    A=[exp(-(x-4)^2 -(y-3)^2) for x in 0:0.1:8, y in 0:0.1:8]
    A[40,40]=-1.0
    @test prod(&,atomic_ratio(A,A,[40.99999999608069, 30.9999999998036],
            [13.102949390292464, 13.10294939029246]) .≈ (1.0, 0.031935091590899516))
end
