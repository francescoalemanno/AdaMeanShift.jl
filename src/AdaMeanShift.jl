module AdaMeanShift

    using Statistics, LinearAlgebra, SpecialFunctions, StaticArrays
    export atomic_meanshift, atomic_intensity, meanshift!, meanshift_nonadaptive!
    @inline function Kern(x::AbstractVector{fT},h::AbstractVector{fT}) where fT
        max(zero(fT),one(fT) - norm(@inbounds x ./ max.(h,one(fT)))^2)
    end
    function clampdim(M,c::T,i)::T where T
        clamp(c,1,size(M,i))
    end
    function makerange(M,p,h,i)
        #round(Int,p[i]-h[i]-1):round(Int,p[i]+h[i]+1)
        @inbounds clampdim(M,round(Int,p[i]-h[i]-1),i):clampdim(M,round(Int,p[i]+h[i]+1),i)
    end



"""
        atomic_meanshift(M,p,h,isotropy,smoothing) -> NamedTuple(pnew,hnew,modeval,delta)

Performs a single iteration of meanshift over a single particle

# Arguments
- `M` density tensor
- `p` initial position for test particle
- `h` initial particle standard deviation
- `isotropy` scalar 0 -> 1, 0 is completely anisotropic, completely isotropic
- `smoothing` scalar 0 -> 1, regularization term for noisy density tensors

"""
    function atomic_meanshift(M,p,h,isotropy,smoothing)
        N=length(p)
        T=eltype(p)
        @assert N==length(h)
        @assert T==eltype(h)
        atomic_meanshift(M,SVector{length(p)}(p),SVector{length(h)}(h),T(isotropy),T(smoothing))
    end
    @generated function atomic_meanshift(M::Mty,p::K,h::K,isotropy::T,smoothing::T) where {T,N,K<:StaticArray,Mty<:AbstractArray{T,N}}
        D=(8//N^2 + 6//N +1)
        scale= sqrt(N * ((2//N + 1)*(4//N + 1)*(6//N + 1))//D)
        invvolume=gamma(N/2+1)*pi^(-N/2)
        modescale=invvolume*N^3*D
        Z=zero(T)
        O=one(T)
        quote
            @inbounds begin
                c=p.*$Z
                xsq=p.*$Z
                NW=IW=KW=W=$Z
                ls=$invvolume*inv(prod(max.(h,$O)))*smoothing

                Base.Cartesian.@nexprs $N i -> R_i = makerange(M,p,h,i)
                Base.Cartesian.@nloops $N i i->R_i begin
                    cp=$K(Base.Cartesian.@ntuple($N,k->i_k))
                    diff=p.-cp
                    keW=Kern(diff,h)
                    imW=Base.Cartesian.@nref($N,M,k->i_k) + ls
                    sW=keW*imW
                    isofactor = isotropy*norm(diff)^2/$N
                    xsq=xsq .+ (diff .^ 2 .*(1.0-isotropy) .+isofactor) .* sW
                    c=c.+cp.*sW
                    W+=sW
                    KW+=keW
                    IW+=imW
                    NW+=$O
                end
                if 2*NW<1
                    return (pnew=p,
                            hnew=h,
                            modeval=T(NaN),
                            delta=$Z)
                end
                xsq=xsq ./ W

                c=@. p+(c/W - p)/2
                return (pnew=K(Base.Cartesian.@ntuple($N,k->clampdim(M,c[k],k))),
                        hnew=max.(1,$scale .* sqrt.(xsq)),
                        modeval=W/prod(max.(h,$O))*$modescale,
                        delta=norm(c.-p))
            end
        end
    end

"""
        atomic_intensity(M,p,h) -> NamedTuple(intensity,numpoints,ndims)

Calculates the integrated intensity on the density tensor `M` over a ellipsoidal region caracterized by `p` region center, `h` semiaxis

# Arguments
- `M` density tensor (e.g. matrix N*N)
- `p` region center (e.g. 2D vector)
- `h` semiaxis lengths (e.g. 2D vector)

Note that both `p`,`h` should be of type Vector or StaticVector with the same length.
"""
    function atomic_intensity(M,p,h)
        N=length(p)
        T=eltype(p)
        @assert N==length(h)
        @assert T==eltype(h)
        atomic_intensity(M,SVector{length(p)}(p),SVector{length(h)}(h))
    end

    @generated function atomic_intensity(M::Mty,p::K,h::K) where {T,N,K<:StaticArray,Mty<:AbstractArray{T,N}}
        Z=zero(T)
        O=one(T)
        quote
            @inbounds begin
                NP=IP=$Z
                Base.Cartesian.@nexprs $N i -> R_i = makerange(M,p,h,i)
                Base.Cartesian.@nloops $N i i->R_i begin
                    cp=$K(Base.Cartesian.@ntuple($N,k->i_k))
                    norm((p.-cp)./h) > 1 && continue;
                    IP+=Base.Cartesian.@nref($N,M,k->i_k)
                    NP+=$O
                end
                return (intensity=IP,numpoints=NP,ndims=N)
            end
        end
    end

using ProgressMeter

    function meanshift_kernel!(::Val{isadaptive} ,M::Mty,P::AbstractVector{K},h::AbstractVector{K},w::AbstractVector{T}, hmax::T, isotropy::T,maxit::T,rtol::T,smoothing::T) where {isadaptive,T,N,K<:StaticArray,Mty<:AbstractArray{T,N}}
        pr = Progress(length(P));
        update!(pr,0)
        l = Threads.SpinLock()
        jj = Threads.Atomic{Int}(0)
        a=@elapsed Threads.@threads for i in eachindex(P)
            cnt=0
            delta=one(T)
            bufcnt=0
            while (cnt<maxit) & (bufcnt < 10)
                if norm(h[i])>hmax && isadaptive
                    h[i]=h[i] ./ norm(h[i]) .* hmax
                end
                data=atomic_meanshift(M,P[i],h[i],isotropy,smoothing)
                if isadaptive
                    @inbounds P[i],h[i],w[i],delta=data
                else
                    @inbounds P[i],_,w[i],delta=data
                end
                cnt+=1
                bufcnt=ifelse(delta<rtol,bufcnt+1,0)
            end
            Threads.lock(l)
            Threads.atomic_add!(jj, 1)
            update!(pr, jj[])
            Threads.unlock(l)
        end
        isadaptive && Threads.@threads for i in eachindex(P)
                if norm(h[i])>hmax
                    h[i]=h[i] ./ norm(h[i]) .* hmax
                end
        end
        finish!(pr)
        a
    end

"""
        meanshift!(M, P, h, w, hmax; isotropy=½, maxit=∞, rtol=√ϵ, smoothing=1)
        meanshift_nonadaptive!(M, P, h, w; maxit=∞, rtol=√ϵ, smoothing=1)

Performs MeanShift on a swarm of particles over a given density matrix locating all modes and their scale. The non adaptive method disables updating of the `h` vectors, and thus depends on fewer parameters.

# Arguments
- `M` is the density tensor over which particles evolve.
- `P` is a julia vector of StaticVectors describing all positions.
- `h` is a julia vector of StaticVectors describing all standard deviations.
- `w` is a julia vector which will contain the new modes intensity estimates.
- `hmax` is a real number for the maximum norm allowed to the `h` vectors.

# Keyword Arguments
- `isotropy` is a scalar [0,1] set to 1 for isotropic kernel, < 1 anisotropy along coordinate axis is allowed.
- `maxit` is a integer for the maximum number of meanshift iterations.
- `rtol` is the absolute tolerance to declare a particle as converged.
- `smoothing` is a regularization term for density tensors with noise.

"""
    function meanshift!(M,P,h,w, hmax; isotropy = 0.5,maxit=Inf,rtol=1e-8,smoothing=1.0)
        meanshift_kernel!(Val(true),M,P,h,w, promote(hmax, isotropy,maxit,rtol,smoothing)...)
    end
    function meanshift_nonadaptive!(M,P,h,w; maxit=Inf,rtol=1e-8,smoothing=1.0)
        meanshift_kernel!(Val(false),M,P,h,w, promote(0, 0,maxit,rtol,smoothing)...)
    end
end
