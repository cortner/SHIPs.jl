using SHIPs, BenchmarkTools
using SHIPs.JacobiPolys: Jacobi
using SHIPs: eval_basis, alloc_B, alloc_dB, alloc_temp, alloc_temp_d




x = 2*rand(50) .- 1
α, β = rand(), rand()
N = 20
J = Jacobi(α, β, N)
P = alloc_B(J, x)
dP = alloc_dB(J, x)
tmp = alloc_temp(J, x)
tmpd = alloc_temp_d(J, x)

SHIPs.JacobiPolys.eval_basis!(P, tmp, J, x)
SHIPs.JacobiPolys.eval_basis_d!(P, dP, tmp, J, x)
@info("Timing for eval_basis!")
@btime SHIPs.JacobiPolys.eval_basis!($P, $tmp, $J, $x)
@btime SHIPs.JacobiPolys.eval_basis!($P, $tmp, $J, $x)
@info("Timing for eval_basis_d!")
@btime SHIPs.JacobiPolys.eval_basis_d!($P, $dP, $tmp, $J, $x)
@btime SHIPs.JacobiPolys.eval_basis_d!($P, $dP, $tmp, $J, $x)


# # Julia Bug?
# @info("Looking at that strange allocation?")
# using BenchmarkTools
# f(A, B) = A, B
# g(A, B) = A
# A = rand(5)
# B = rand(5)
# @btime f($A, $B);
# @btime f(1, 2);
# @btime g($A, $B);
#
# function runn(P, dP, J, x, N)
#    for n = 1:1000
#       SHIPs.JacobiPolys.eval_basis_d!(P, dP, J, rand(), N)
#    end
#    return nothing
# end
#
# @info("Try again inside a function - no allocation!")
# @btime runn($P, $dP, $J, $x, 15)
