
using SHIPs, JuLIP, BenchmarkTools
using JuLIP: alloc_temp, alloc_temp_d

trans = PolyTransform(2, 1.0)
fcut = PolyCutoff2s(Val(2), 0.5, 3.0)
ships = [SHIPBasis(SparseSHIP(n, 15, 1.5), trans, fcut)
         for n = 2:4]

NR = 50
Rs = 1.0 .+ 2*(rand(JVecF, NR) .- 0.5)
Zs = zeros(Int16, NR)
z0 = 0

@info("profile `precompute_A!`")
tmp = alloc_temp(ships[1], NR)
@btime SHIPs.precompute_A!($tmp, $(ships[1]), $Rs, $Zs)
@btime SHIPs.precompute_A!($tmp, $(ships[1]), $Rs, $Zs)
@info("profile `precompute_grads!`")
tmpd = alloc_temp_d(ships[1], NR)
@btime SHIPs.precompute_grads!($tmpd, $(ships[1]), $Rs, $Zs)
@btime SHIPs.precompute_grads!($tmpd, $(ships[1]), $Rs, $Zs)


@info("profile basis computation")
for n = 2:4
   @info("  body-order $(n+1):")
   🚢 = ships[n-1]
   B = SHIPs.alloc_B(🚢)
   @info("     eval_basis:")
   @btime SHIPs.eval_basis!($B, $tmp, $🚢, $Rs, $Zs, $z0)
   @btime SHIPs.eval_basis!($B, $tmp, $🚢, $Rs, $Zs, $z0)
   @info("     eval_basis_d:")
   dB = SHIPs.alloc_dB(🚢, Rs)
   @btime SHIPs.eval_basis_d!($B, $dB, $tmpd, $🚢, $Rs, $Zs, $z0)
   @btime SHIPs.eval_basis_d!($B, $dB, $tmpd, $🚢, $Rs, $Zs, $z0)
end

# ##
# using Profile
# 🚢 = ships[2]
# B = SHIPs.alloc_B(🚢)
# dB = SHIPs.alloc_dB(🚢, Rs)
# @btime SHIPs.eval_basis!($B, $tmp, $🚢, $Rs, $Zs, $z0)
# @btime SHIPs.eval_basis_d!($B, $dB, $tmpd, $🚢, $Rs, $Zs, $z0)
#
# @code_warntype SHIPs._eval_basis!(B, tmp, 🚢, Val{3}(), 1, 🚢.NuZ[3,1])
#
# SHIPs._eval_basis!(B, tmp, 🚢, Val{3}(), 1, 🚢.NuZ[3,1])
#
# ##
#
# function runn(N, f, args...)
#    for n = 1:N
#       f(args...)
#    end
# end
#
# runn(10, SHIPs.eval_basis!, B, tmp, 🚢, Rs, Zs, z0)
# runn(10, SHIPs.eval_basis_d!, B, dB, tmpd, 🚢, Rs, Zs, z0)
#
# ##
#
# Profile.clear()
# @profile runn(100, SHIPs.eval_basis!, B, tmp, 🚢, Rs, Zs, z0)
# Profile.print()
