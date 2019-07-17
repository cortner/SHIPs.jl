
module JacobiPolys

import SHIPs: eval_basis,
              eval_basis!,
              eval_basis_d!,
              alloc_B,
              alloc_dB

import JuLIP: alloc_temp, alloc_temp_d

import Base.==
export Jacobi

"""
`Jacobi{T} : ` represents the basis of Jacobi polynomials
parameterised by α, β up to some fixed maximum degree. Recall that Jacobi
polynomials are orthogonal on [-1,1] w.r.t. the weight
w(x) = (1-x)^α (1+x)^β.

### Constructor:
```
Jacobi(α, β, N)   # N = max degree
```

### Evaluate basis and derivatives:
```
x = 2*(rand() - 0.5)
P = zeros(N)
eval_basis!(P, J, x, N)
dP = zeros(N)
eval_basis_d!(P, dP, J, x, N)   # evaluates both P, dP
```

### Notes

`Jacobi(...)` precomputes the recursion coefficients using arbitrary
precision arithmetic, then stores them as `Vector{Float64}`. The recursion
is then given by
```
P_{n} = (A[n] * x + B[n]) * P_{n-1} + C[n] * P_{n-2}
```
"""
struct Jacobi{T}
   α::T
   β::T
   A::Vector{T}
   B::Vector{T}
   C::Vector{T}
end

==(J1::Jacobi, J2::Jacobi) = (
      (J1.α == J2.α) && (J1.β == J2.β) && (length(J1) == length(J2))
   )


function Jacobi(α, β, N, T=Float64)
   # precompute the recursion coefficients
   A = zeros(T, N)
   B = zeros(T, N)
   C = zeros(T, N)
   for n = 2:N
      c1 = big(2*n*(n+α+β)*(2*n+α+β-2))
      c2 = big(2*n+α+β-1)
      A[n] = T( big(2*n+α+β)*big(2*n+α+β-2)*c2 / c1 )
      B[n] = T( big(α^2 - β^2) * c2 / c1 )
      C[n] = T( big(-2*(n+α-1)*(n+β-1)*(2n+α+β)) / c1 )
   end
   return Jacobi(T(α), T(β), A, B, C)
end


Base.length(J::Jacobi) = maxdegree(J) + 1
maxdegree(J::Jacobi) = length(J.A)
alloc_B(J::Jacobi{T}, x::AbstractVector) where {T} =
      zeros(T, length(x), length(J))
alloc_dB(J::Jacobi{T}, x::AbstractVector) where {T} =
      zeros(T, length(x), length(J))


function eval_basis!(P::AbstractMatrix, tmp, J::Jacobi, x::AbstractVector)
   N = maxdegree(J)
   @assert size(P, 2) >= N+1
   @assert size(P, 1) >= length(x)
   @assert N >= 2
   α, β = J.α, J.β
   @inbounds begin
      for j = 1:length(x)
         P[j, 1] = 1
         P[j, 2] = (α+1) + 0.5 * (α+β+2) * (x[j]-1)
      end
      for n = 2:N
         A, B, C = J.A[n], J.B[n], J.C[n]
         for j = 1:length(x)
            P[j, n+1] = (A * x[j] + B) * P[j, n] + C * P[j, n-1]
         end
      end
   end
   return P
end

# # mostly for testing and debugging
# eval_basis(J::Jacobi, x::Number) =
#       eval_basis!(zeros(1, maxdegree(J)+1), J, [x], nothing)[:]


function eval_basis_d!(P::AbstractMatrix, dP::AbstractMatrix, tmp,
                       J::Jacobi, x::AbstractVector)
   N = maxdegree(J)
   @assert length(P) >= N+1
   @assert length(dP) >= N+1
   @assert N >= 2
   α, β = J.α, J.β
   @inbounds begin
      for j = 1:length(x)
         P[j, 1] = 1
         dP[j, 1] = 0
         P[j, 2] = (α+1) + 0.5 * (α+β+2) * (x[j]-1)
         dP[j, 2] = 0.5 * (α+β+2)
      end
      for n = 2:N
         A, B, C = J.A[n], J.B[n], J.C[n]
         for j = 1:length(x)
            c1 = A * x[j] + B
            P[j, n+1] = c1 * P[j, n] + C * P[j, n-1]
            dP[j, n+1] = A * P[j, n] + c1 * dP[j, n] + C * dP[j, n-1]
         end
      end
   end
   # return P, dP
   return nothing
end


end
