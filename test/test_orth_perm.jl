
@testset "Basis Orthogonality under permutation" begin

##
using Test
using SHIPs, JuLIP, JuLIP.Testing, QuadGK, LinearAlgebra, SHIPs.JacobiPolys
using SHIPs: TransformedJacobi, transform, transform_d, eval_basis!,
             alloc_B, alloc_temp
using StaticArrays
using Combinatorics

##
@info("Testing orthogonality of permutation-invariant Ylm via sampling")

function scalar_prod(L1,M1,L2,M2,SH,N,Nsamples = 1_000_000)
   res = 0.
   I1 = [SHIPs.SphericalHarmonics.index_y(L1[i],M1[i]) for i in 1:N]
   I2 = [SHIPs.SphericalHarmonics.index_y(L2[i],M2[i]) for i in 1:N]
   for n=1:Nsamples
      # generate samples
      Rs = [SHIPs.Utils.rand_sphere() for i=1:N]
      Y = [SHIPs.eval_basis(SH, Rs[i]) for i=1:N]

      # Sum over permutations
      BL1M1 = sum(prod(Y[σ[i]][I1[i]] for i in 1:N) for σ in permutations(1:N))
      BL2M2 = sum(prod(Y[σ[i]][I2[i]] for i in 1:N) for σ in permutations(1:N))
      res += BL1M1*BL2M2
   end
   return res / Nsamples
end


function scalar_prod(LM1,LM2,SH,N,Nsamples = 1_000_000)
   res = 0.
   I1 = [SHIPs.SphericalHarmonics.index_y(LM1[i][1],LM1[i][2]) for i in 1:N]
   I2 = [SHIPs.SphericalHarmonics.index_y(LM2[i][1],LM2[i][2]) for i in 1:N]
   for n=1:Nsamples
      # generate samples
      Rs = [SHIPs.Utils.rand_sphere() for i=1:N]
      Y = [SHIPs.eval_basis(SH, Rs[i]) for i=1:N]

      # Sum over permutations
      BL1M1 = sum(prod(Y[σ[i]][I1[i]] for i in 1:N) for σ in permutations(1:N))
      BL2M2 = sum(prod(Y[σ[i]][I2[i]] for i in 1:N) for σ in permutations(1:N))
      res += BL1M1*BL2M2
   end
   return res / Nsamples
end



function rand_test(maxL,N,Nsamples = 1_000_000)
   SH = SHIPs.SphericalHarmonics.SHBasis(maxL)
   L1 =  rand(collect(0:maxL),N)
   L2 =  rand(collect(0:maxL),N)
   M1 = [rand(-L1[i]:L1[i]) for i in 1:N]
   M2 = [rand(-L2[i]:L2[i]) for i in 1:N]
   r = scalar_prod(L1,M1,L2,M2,SH,N,Nsamples)
   @show L1,L2,M1,M2,r
end

function rand_test(maxL,N,Nsamples = 1_000_000)
   SH = SHIPs.SphericalHarmonics.SHBasis(maxL)
   LM1 = Tuple{Int64,Int64}[]
   LM2 = Tuple{Int64,Int64}[]
   L = collect(0:maxL)
   for i = 1:N
      l1 = rand(L)
      l2 = rand(L)
      m1 = rand(-l1:l1)
      m2 = rand(-l2:l2)
      push!(LM1,(l1,m1))
      push!(LM2,(l2,m2))
   end
   r = scalar_prod(LM1,LM2,SH,N,Nsamples)
   @show LM1,LM2,r
end

rand_test(4,2)



# function gram_matrix(Llist,Mlist,Nsamples = 10)
#    G = zeros(Complex,length(Llist),length(Llist))
#
#    for n=1:Nsamples
#       # generate samples
#       Rs = [SHIPs.Utils.rand_sphere() for i=1:N]
#       Y = [SHIPs.eval_basis(SH, Rs[i]) for i=1:N]
#
#       for l1 in 1:length(Llist)
#          for l2 in 1:length(Llist)
#             I1 = [SHIPs.SphericalHarmonics.index_y(
#                   Llist[l1][i],Mlist[l1][i]) for i in 1:N]
#             I2 = [SHIPs.SphericalHarmonics.index_y(
#                   Llist[l2][i],Mlist[l2][i]) for i in 1:N]
#             # Sum over permutations
#             BL1M1 = sum(prod(Y[σ[i]][I1[i]] for i in 1:N)
#                         for σ in permutations(1:N))
#             BL2M2 = sum(prod(Y[σ[i]][I2[i]] for i in 1:N)
#                         for σ in permutations(1:N))
#             G[l1,l2] += BL1M1*BL2M2
#          end
#       end
#    end
#    return G ./ Nsamples
# end


# Llist = []
# Mlist = []
# for l1 in 0:maxL
#    for m1 in -l1:l1
#       for l2 in l1:maxL
#          for m2 in -l2:l2
#             for l3 in l2:maxL
#                for m3 in -l3:l3
#                   global Llist
#                   global Mlist
#                   push!(Llist,[l1,l2,l3])
#                   push!(Mlist,[m1,m2,m3])
#                end
#             end
#          end
#       end
#    end
# end
#
# Llist
# Mlist
#
# gram_matrix(Llist,Mlist)




end
