module SHIPs

const IntS = Int32

using Reexport
@reexport using JuLIP

include("aux.jl")
include("prototypes.jl")
include("jacobi.jl")
include("sphericalharmonics.jl")
include("transforms.jl")

# in here we specify body-order specific code so that it doesn't pollute
# the main codebase
#  * filter_tuples
#  * _Bcoeff
include("bodyorders.jl")

# basis specification - which basis functions to keep
include("degrees.jl")

# actual basis implementation 
include("basis.jl")
include("fast.jl")
include("pair.jl")

end # module
