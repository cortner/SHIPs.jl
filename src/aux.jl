
using StaticArrays
using SparseArrays: SparseMatrixCSC

# -----------------------------------
# iterating over an m collection
# -----------------------------------

_mvec(::CartesianIndex{0}) = SVector(IntS(0))

_mvec(mpre::CartesianIndex) = SVector(Tuple(mpre)..., - sum(Tuple(mpre)))

struct MRange{N, TI, T2}
   ll::SVector{N, TI}
   cartrg::T2
end

Base.length(mr::MRange) = sum(_->1, _mrange(mr.ll))

"""
Given an l-vector `ll` iterate over all combinations of `mm` vectors  of
the same length such that `sum(mm) == 0`
"""
_mrange(ll) = MRange(ll, Iterators.Stateful(
                        CartesianIndices(ntuple(i -> -ll[i]:ll[i], length(ll)-1))))

function Base.iterate(mr::MRange{1, TI}, args...) where {TI}
   if isempty(mr.cartrg)
      return nothing
   end
   while !isempty(mr.cartrg)
      popfirst!(mr.cartrg)
   end
   return SVector{1, TI}(0), nothing
end

function Base.iterate(mr::MRange, args...)
   while true
      if isempty(mr.cartrg)
         return nothing
      end
      mpre = popfirst!(mr.cartrg)
      if abs(sum(mpre.I)) <= mr.ll[end]
         return _mvec(mpre), nothing
      end
   end
   error("we should never be here")
end


# _mrange_prefilter(ll) = (
#                   _mvec(mpre) for mpre in CartesianIndices(
#                                 ntuple(i -> -ll[i]:ll[i], length(ll)-1) )
#               )




"""
`@generated function nfcalls(::Val{N}, f)`

Effectively generates a loop of functions calls, but fully unrolled and
therefore type-stable:
```{julia}
f(Val(1))
f(Val(2))
f(Val(3))
# ...
f(Val(N))
```
"""
@generated function nfcalls(::Val{N}, f) where {N}
   code = Expr[]
   for n = 1:N
      push!(code, :(f(Val($n))))
   end
   quote
      $(Expr(:block, code...))
      return nothing
   end
end


"""
`@generated function valnmapreduce(::Val{N}, v, f)`

Generates a map-reduce like code, with fully unrolled loop which makes this
type-stable,
```{julia}
begin
   v += f(Val(1))
   v += f(Val(2))
   # ...
   v += f(Val(N))
   return v
end
```
"""
@generated function valnmapreduce(::Val{N}, v, f) where {N}
   code = Expr[]
   for n = 1:N
      push!(code, :(v += f(Val($n))))
   end
   quote
      $(Expr(:block, code...))
      return v
   end
end


# (de-)serialize a sparse matric
Base.Dict(A::SparseMatrixCSC) =
   Dict("__id__" => "SparseMatrixCSC",
        "colptr" => A.colptr,
        "rowval" => A.rowval,
        "nzval" => A.nzval,
        "m" => m,
        "n" => n )

Base.convert(::Val{:SparseMatrixCSC}, D::Dict) =
   SparseMatrixCSC(D)

SparseMatrixCSC(D::Dict) =
   SparseMatrixCSC(D["m"], D["n"], D["colptr"], D["rowval"], D["nzval"])
