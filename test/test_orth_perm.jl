
@testset "Basis Orthogonality under permutation" begin

##
using Test
using SHIPs, JuLIP, JuLIP.Testing, QuadGK, LinearAlgebra, SHIPs.JacobiPolys
using SHIPs: TransformedJacobi, transform, transform_d, eval_basis!,
             alloc_B, alloc_temp, _mrange, _get_ll
using SHIPs.Rotations: CoeffArray, basis
using StaticArrays
using Combinatorics

##
@info("Testing orthogonality of permutation-invariant Ylm via sampling")


function scalar_prod(LM1,LM2,SH,Nsamples = 1_000_000)
   @assert length(LM1) == length(LM2)
   N = length(LM1)
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
      res += BL1M1*BL2M2'
   end
   return res / Nsamples
end

#
# function scalar_prod_without_perm(LM1,LM2,SH,N,Nsamples = 1_000_000)
#    res = 0.
#    I1 = [SHIPs.SphericalHarmonics.index_y(LM1[i][1],LM1[i][2]) for i in 1:N]
#    I2 = [SHIPs.SphericalHarmonics.index_y(LM2[i][1],LM2[i][2]) for i in 1:N]
#    for n=1:Nsamples
#       # generate samples
#       Rs = [SHIPs.Utils.rand_sphere() for i=1:N]
#       Y = [SHIPs.eval_basis(SH, Rs[i]) for i=1:N]
#
#       # Sum over permutations
#       BL1M1 = sum(prod(Y[σ[i]][I1[i]] for i in 1:N) for σ in permutations(1:N))
#       BL2M2 = sum(prod(Y[σ[i]][I2[i]] for i in 1:N) for σ in permutations(1:N))
#       res += BL1M1*BL2M2
#    end
#    return res / Nsamples
# end


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
   if sort(LM1) != sort(LM2)
      return scalar_prod(LM1,LM2,SH,Nsamples)
   else
      return 0.
   end
end


function _get_loop_ex(N,maxL)
   # generate the expression for the multiple for loops, e.g. for 4B:
   # for i_1 = 1:maxL, i_2 = i_1:maxL, i_3 = i_2:maxL
   str_loop = "for i_1 = 1:($maxL)"
   if N>1
      for n = 2:N
         str_loop *= ", i_$n = (i_$(n-1)):($maxL)"
      end
   end
   str_loop *= "\n end"
   return Meta.parse(str_loop)
end


function _get_Jvec_ex(N)
   # inside these N-1 loops we collect the loop indices into an SVector, e.g.
   # J = @SVector [i_1, i_2, i_3]
   str = "J = @SVector Int[i_1"
   for n = 2:N
      str *= ", i_$n"
   end
   return Meta.parse(str * "]")
end

@generated function generateL(::Val{N},::Val{maxL}) where {N,maxL}
   code = Expr[]
   # initialise the output
   push!(code, :( ll = []  ))
   # generate the multi-for-loop
   ex_loop = _get_loop_ex(N,maxL)
   # inside the loop
   # ---------------
   code_inner = Expr[]
   # collect the indices into a vector
   push!(code_inner,      _get_Jvec_ex(N) )
   push!(code_inner, :(   push!(ll,J) )   )
   # put code_inner into the loop expression
   ex_loop.args[2] = Expr(:block, code_inner...)
   # now append the loop to the main code
   push!(code, ex_loop)
   quote
      @inbounds $(Expr(:block, code...))
      return ll
   end
end


function generateLM(N,maxL)
   LL = generateL(Val(N),Val(maxL))
   LM = []
   for ll in LL
      MM = collect(_mrange(ll))
      lm = []
      for mm in MM
         lm = [(ll[i],mm[i]) for i in 1:N]
         if issorted(lm)&&(sum(mm)==0)
            push!(LM,lm)
         end
      end
   end
   return LM
end



function gramian(maxL,N,Nsamples = 10_000)
   SH = SHIPs.SphericalHarmonics.SHBasis(maxL)
   LM = generateLM(N,maxL)
   nb = length(LM)
   @show nb
   G = zeros(ComplexF64,nb,nb)
   for i=1:nb, j=1:nb
      G[i,j] = scalar_prod(LM[i],LM[j],SH,Nsamples)
   end
   return G
end

maxL = 10
N = 1
LL = generateL(Val(N),Val(maxL))


generateL(Val(N),Val(maxL))

generateLM(2,2)

G = gramian(3,2,100_000)
cond(G)
abs.(G)

@info("Testing orthogonality of permutation-invariant
       and rotation-invariant Ylm via sampling")

function scalar_prod(L1,L2,SH,Nsamples = 1_000_000)
    @assert length(L1) == length(L2)
    N = length(L1)
    U1 = basis(CoeffArray(), L1)
    J1 = size(U1,2)
    U2 = basis(CoeffArray(), L2)
    J2 = size(U2,2)
    res = zeros(ComplexF64,J1,J2)
    I1 = [SHIPs.SphericalHarmonics.index_y(LM1[i][1],LM1[i][2]) for i in 1:N]
    I2 = [SHIPs.SphericalHarmonics.index_y(LM2[i][1],LM2[i][2]) for i in 1:N]
    for n=1:Nsamples
        # generate samples
        Rs = [SHIPs.Utils.rand_sphere() for i=1:N]
        Y = [SHIPs.eval_basis(SH, Rs[i]) for i=1:N]
        # Sum over permutations
        BL1M1 = sum(prod(Y[σ[i]][I1[i]] for i in 1:N) for σ in permutations(1:N))
        BL2M2 = sum(prod(Y[σ[i]][I2[i]] for i in 1:N) for σ in permutations(1:N))
        res += BL1M1*BL2M2'
    end
    return res / Nsamples
end





ll = SVector(1,1,1,2)




#
# LM1 = [(1,-1), (1,0), (1,0)]
# SH = SHIPs.SphericalHarmonics.SHBasis(1)
# scalar_prod(LM1,LM1,SH,10_000)
#
#
# @test abs.(rand_test(4,2))<1e-4
#



# trans = PolyTransform(2, 1.0)
# cutf = PolyCutoff2s(2, 0.5, 3.0)
#
# ship2 = SHIPBasis(SparseSHIP(2, 15; wL = 2.0), trans, cutf)
# ship3 = SHIPBasis(SparseSHIP(3, 13; wL = 2.0), trans, cutf)
# ship4 = SHIPBasis(SparseSHIP(4, 10; wL = 1.5), trans, cutf)
# ship5 = SHIPBasis(SparseSHIP(5,  8; wL = 1.5), trans, cutf)
# ship6 = SHIPBasis(SparseSHIP(6,  8; wL = 1.5), trans, cutf)
# ships = [ship2, ship3, ship4, ship5, ship6]
#
# ship2.KL[1][1]
# i = 14
#
# for
#
# ll = [_get_ll(ship2.KL[1], ship2.NuZ[2][i]) for i in 1:length(ship2.NuZ[2])]
#
# collect(_mrange(SVector(1,2,3)))
#
#
# length(ship2.KL)
# size(ship2.NuZ)
#
# _get_ll(B.KL[2], B.NuZ[2,2][5])
#
# ship2.KL
# ship2.NuZ[2]
# ship2.NuZ[2][1].ν
#
# B.Nuz[1,1][1]



# function rand_test(maxL,N,Nsamples = 1_000_000)
#    SH = SHIPs.SphericalHarmonics.SHBasis(maxL)
#    L1 =  rand(collect(0:maxL),N)
#    L2 =  rand(collect(0:maxL),N)
#    M1 = [rand(-L1[i]:L1[i]) for i in 1:N]
#    M2 = [rand(-L2[i]:L2[i]) for i in 1:N]
#    r = scalar_prod(L1,M1,L2,M2,SH,N,Nsamples)
#    @show L1,L2,M1,M2,r
# end


# function scalar_prod(L1,M1,L2,M2,SH,N,Nsamples = 1_000_000)
#    res = 0.
#    I1 = [SHIPs.SphericalHarmonics.index_y(L1[i],M1[i]) for i in 1:N]
#    I2 = [SHIPs.SphericalHarmonics.index_y(L2[i],M2[i]) for i in 1:N]
#    for n=1:Nsamples
#       # generate samples
#       Rs = [SHIPs.Utils.rand_sphere() for i=1:N]
#       Y = [SHIPs.eval_basis(SH, Rs[i]) for i=1:N]
#
#       # Sum over permutations
#       BL1M1 = sum(prod(Y[σ[i]][I1[i]] for i in 1:N) for σ in permutations(1:N))
#       BL2M2 = sum(prod(Y[σ[i]][I2[i]] for i in 1:N) for σ in permutations(1:N))
#       res += BL1M1*BL2M2
#    end
#    return res / Nsamples
# end


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
