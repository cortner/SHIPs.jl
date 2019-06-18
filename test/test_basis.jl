
using SHIPs, JuLIP, BenchmarkTools

trans = PolyTransform(2, 1.0)
ship = SHIPBasis(3, 15, 2.0, trans, 2, 0.5, 3.0
   )

Rs = 1.0 .+ rand(JVecF, 100)
@btime SHIPs.precompute_A!($ship, $Rs)
SHIPs.length_B(ship)
length(ship.A)

@code_warntype SHIPs.precompute_A!(ship, Rs)

using Profile
##
function runn(ship, Rs, N)
   for n = 1:N
      SHIPs.precompute_A!(ship, Rs)
   end
   return ship
end
Profile.clear()
runn(ship, Rs, 10)
@profile runn(ship, Rs, 10_000)
Profile.print()
