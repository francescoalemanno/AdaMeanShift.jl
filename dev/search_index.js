var documenterSearchIndex = {"docs":
[{"location":"#AdaMeanShift.jl-1","page":"Home","title":"AdaMeanShift.jl","text":"","category":"section"},{"location":"#","page":"Home","title":"Home","text":"Julia package to perform multidimensional and scale-adaptive meanshift over density tensors (Images, Histograms, ...)","category":"page"},{"location":"#","page":"Home","title":"Home","text":"","category":"page"},{"location":"#","page":"Home","title":"Home","text":"Modules = [AdaMeanShift]","category":"page"},{"location":"#AdaMeanShift.atomic_intensity-Tuple{Any,Any,Any}","page":"Home","title":"AdaMeanShift.atomic_intensity","text":"    atomic_intensity(M::Mty,p::K,h::K) where {T,N,K<:StaticArray,Mty<:AbstractArray{T,N}}\n\nCalculates the integrated intensity on the density tensor M over a ellipsoidal region caracterized by p region center, h semiaxis\n\nArguments\n\nM density tensor\np region center\nh semiaxis lengths\n\nNote that both p,h should be of type Vector or StaticVector with the same length.\n\n\n\n\n\n","category":"method"},{"location":"#AdaMeanShift.atomic_meanshift-NTuple{5,Any}","page":"Home","title":"AdaMeanShift.atomic_meanshift","text":"    atomic_meanshift(M,p,h,isotropy,smoothing) -> NamedTuple(pnew,hnew,modeval,delta)\n\nPerforms a single iteration of meanshift over a single particle\n\nArguments\n\nM density tensor\np initial position for test particle\nh initial particle standard deviation\nisotropy scalar 0 -> 1, 0 is completely anisotropic, completely isotropic\nsmoothing scalar 0 -> 1, regularization term for noisy density tensors\n\n\n\n\n\n","category":"method"},{"location":"#AdaMeanShift.meanshift!-Union{Tuple{Mty}, Tuple{K}, Tuple{N}, Tuple{T}, Tuple{Mty,AbstractArray{K,1},AbstractArray{K,1},AbstractArray{T,1},T}} where Mty<:AbstractArray{T,N} where K<:StaticArrays.StaticArray where N where T","page":"Home","title":"AdaMeanShift.meanshift!","text":"    meanshift!(M, P, h, w, hmax; isotropy=½, maxit=∞, rtol=√ϵ, smoothing=1)\n\nPerforms MeanShift on a swarm of particles over a given density matrix locating all modes and their scale.\n\nArguments\n\nM is the density tensor over which particles evolve.\nP is a julia vector of StaticVectors describing all positions.\nh is a julia vector of StaticVectors describing all standard deviations.\nw is a julia vector which will contain the new modes intensity estimates.\n\nKeyword Arguments\n\nisotropy is a scalar [0,1] set to 1 for isotropic kernel, < 1 anisotropy along coordinate axis is allowed.\nmaxit is a integer for the maximum number of meanshift iterations.\nrtol is the absolute tolerance to declare a particle as converged.\nsmoothing is a regularization term for density tensors with noise.\n\n\n\n\n\n","category":"method"}]
}
