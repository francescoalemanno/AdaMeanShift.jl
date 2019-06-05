var documenterSearchIndex = {"docs":
[{"location":"#AdaMeanShift.jl-1","page":"Home","title":"AdaMeanShift.jl","text":"","category":"section"},{"location":"#","page":"Home","title":"Home","text":"Julia package to perform multidimensional and scale-adaptive meanshift over density tensors (Images, Histograms, ...)","category":"page"},{"location":"#","page":"Home","title":"Home","text":"","category":"page"},{"location":"#","page":"Home","title":"Home","text":"Modules = [AdaMeanShift]","category":"page"},{"location":"#AdaMeanShift.runms!-Union{Tuple{Mty}, Tuple{K}, Tuple{N}, Tuple{T}, Tuple{Mty,AbstractArray{K,1},AbstractArray{K,1},AbstractArray{T,1},T}, Tuple{Mty,AbstractArray{K,1},AbstractArray{K,1},AbstractArray{T,1},T,T}, Tuple{Mty,AbstractArray{K,1},AbstractArray{K,1},AbstractArray{T,1},T,T,T}, Tuple{Mty,AbstractArray{K,1},AbstractArray{K,1},AbstractArray{T,1},T,T,T,T}} where Mty<:AbstractArray{T,N} where K<:StaticArrays.StaticArray where N where T","page":"Home","title":"AdaMeanShift.runms!","text":"runms!(M,P,h,w,hmax,isotropy,maxit,rtol) –– Performs MeanShift on a swarm of particles over a given density matrix      locating all modes and their scale.\n\n* M is the density matrix over which particles evolve\n* P is a julia vector of StaticVectors describing all positions\n* h is a julia vector of StaticVectors describing all standard deviations\n* w is a julia vector which will contain the new modes intensity estimates\n* hmax is the maximum norm that standard deviations \"h\" can have\n* isotropy=0.5 is a scalar [0,1] set to 1 for isotropic kernel,\n                   < 1 anisotropy along coordinate axis is allowed\n* maxit=Inf is a integer for the maximum number of meanshift iterations\n* rtol=√ϵ is the absolute tolerance to declare a particle as converged\n\n\n\n\n\n","category":"method"}]
}
