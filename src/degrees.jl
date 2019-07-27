
export TotalDegree, HyperbolicCross

abstract type AbstractDegree{BO} end

VecOrTup = Union{AbstractVector, Tuple}

admissible(D::AbstractDegree, k, l) = deg(D, k, l) <= D.deg

deg(D::AbstractDegree, k::Integer, l::Integer) =
      k + D.wL * l

"""
`TotalDegree` : a sparse-grid type degree definition,
```
deg({k}, {l}) = ∑ (k + wL * l)
```
"""
struct TotalDegree{BO} <: AbstractDegree{BO}
   deg::Int
   wL::Float64
   valBO::Val{BO}
end

TotalDegree(bo::Integer, deg::Integer, wL::Float64) =
      TotalDegree(deg, wL, Val(bo))

deg(D::TotalDegree, kk::VecOrTup, ll::VecOrTup) =
      sum( deg(D, k, l) for (k, l) in zip(kk, ll) )
maxK(D::TotalDegree) = D.deg
maxL(D::TotalDegree) = floor(Int, D.deg / D.wL)
maxL(D::TotalDegree, k::Integer) = floor(Int, (D.deg - k) / D.wL)

bodyorder(Deg::AbstractDegree{BO}) where {BO} = BO

Dict(D::TotalDegree) = Dict("__id__" => "SHIPs_TotalDegree",
                            "deg" => D.deg, "wL" => D.wL,
                            "bo" => bodyorder(D))
convert(::Val{:SHIPs_TotalDegree}, D::Dict) =
      TotalDegree(D["deg"], D["wL"], B["bo"])


# """
# `HyperbolicCross` : standard hyperbolic cross degree,
# ```
# deg({k}, {l}) = prod( max(1, k + wL * l) )
# ```
# """
# struct HyperbolicCross <: AbstractDegree
#    deg::Int
#    wL::Float64
# end
#
# deg(D::HyperbolicCross, kk::VecOrTup, ll::VecOrTup) =
#       prod( max(1, deg(D, k, l)) for (k, l) in zip(kk, ll) )
# maxK(D::HyperbolicCross) = D.deg
# maxL(D::HyperbolicCross) = floor(Int, D.deg / D.wL)
# maxL(D::HyperbolicCross, k::Integer) = floor(Int, (D.deg - k) / D.wL)
#
# Dict(D::HyperbolicCross) = Dict("__id__" => "SHIPs_HyperbolicCross",
#                             "deg" => D.deg, "wL" => D.wL)
# convert(::Val{:SHIPs_HyperbolicCross}, D::Dict) =
#       HyperbolicCross(D["deg"], D["wL"])




function generate_KL(D::AbstractDegree, TI = IntS, TF=Float64)
   allKL = NamedTuple{(:k, :l, :deg), Tuple{TI,TI,TF}}[]
   degs = TF[]
   # morally "k + wL * l <= deg"
   for k = 0:maxK(D), l = 0:maxL(D, k)
      push!(allKL, (k=k, l=l, deg=deg(D, k, l)))
      push!(degs, deg(D, k, l))
   end
   # sort allKL according to total degree
   I = sortperm(degs)
   return allKL[I], degs[I]
end




_mrange(ll::SVector{BO}) where {BO} =
   CartesianIndices(ntuple( i -> -ll[i]:ll[i], (BO-1) ))

"""
return kk, ll, mrange
where kk, ll is BO-tuples of k and l indices, while mrange is a
cartesian range over which to iterate to construct the basis functions

(note: this is tested for correcteness and speed)
"""
function _klm(ν::StaticVector{BO, T}, KL) where {BO, T}
   kk = SVector( ntuple(i -> KL[ν[i]].k, BO) )
   ll = SVector( ntuple(i -> KL[ν[i]].l, BO) )
   mrange = CartesianIndices(ntuple( i -> -ll[i]:ll[i], (BO-1) ))
   return kk, ll, mrange
end


# TODO [tuples] generate basis functions UP TO A BODY-ORDER
#      we need to have 0 stand for a 1 (i.e. body-order)
#      => decide whether to drop T0.

"""
create a vector of Nu arrays with the right type information
for each body-order
"""
function _generate_Nu(bo::Integer, T=IntS)
   Nu = []
   for n = 1:bo
      push!(Nu, SVector{n, T}[])
   end
   # convert into an SVector to make the length a type parameters
   return SVector(Nu...)
end

function generate_KL_tuples(Deg::AbstractDegree, cg; filter=true)
   maxbo = bodyorder(Deg)
   # all possible (k, l) pairs
   allKL, degs = generate_KL(Deg)
   # sepatare arrays for all body-orders
   Nu = _generate_Nu(maxbo)
   for N = 1:maxbo
      _generate_KL_tuples!(Nu[N], Deg, cg, allKL, degs; filter=filter)
   end
   return allKL, Nu
end

function _generate_KL_tuples!(Nu::Vector{<: SVector{BO}}, Deg::AbstractDegree,
                             cg, allKL, degs; filter=true) where {BO}
   # the first iterm is just (0, ..., 0)
   # we can choose (k1, l1), (k2, l2) ... by indexing into allKL
   # then we start incrementing until we hit the maximum degree
   # while retaining the ordering ν₁ ≤ ν₂ ≤ …
   lastidx = 0
   ν = @MVector ones(IntS, BO)   # (ones(IntS, bo)...)
   while true
      # check whether the current ν tuple is admissible
      # the first condition is that its max index is small enough
      isadmissible = maximum(ν) <= length(allKL)
      if isadmissible
         # the second condition is that the multivariate degree it defines
         # is small enough => for that we first have to compute the corresponding
         # k and l vectors
         kk, ll, _ = _klm(ν, allKL)
         isadmissible = admissible(Deg, kk, ll)
      end

      # we want to increment `curindex`, but if we've reach the maximum degree
      # then we need to move to the next index down

      # if the current tuple ν has admissible degree ...
      if isadmissible
         # ... then we add it to the stack  ...
         #     (at least if it is an admissible basis function respecting
         #      all the symmetries - this is checked by filter_tuples)
         if !filter || filter_tuples(allKL, ν, cg)
            push!(Nu, SVector(ν))
         end
         # ... and increment it
         lastidx = BO
         ν[lastidx] += 1
      else
         # we have overshot, _deg(ν) > deg; we must go back down, by
         # decreasing the index at which we increment
         if lastidx == 1
            break
         end
         ν[lastidx-1:end] .= ν[lastidx-1] + 1
         lastidx -= 1
      end
   end
   return allKL, Nu
end
