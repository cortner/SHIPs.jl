
# first load the correct Julia Binary.
# (on my system pyjulia never finds it by itself for some reason...)
from julia import Julia
jl = Julia(runtime="julia")
# import `Main` since this is where Julia loads all new modules
from julia import Main
# load the SHIPDescriptors module into Main
jl.include("SHIPsDescriptors.jl")

# load anything else you need
from ase.build import bulk

Main.at = bulk("Si", cubic=True) * (2,2,2)
# this generates a new descriptor; make sure to store this in `Main` (i.e.
# inside of Julia) - see help text to see the complete set of options
Main.eval("desc = SHIPsDescriptors.SHIPDescriptor(at; deg=6, rcut=4.0)");
# create an ase.Atoms object by whatever means
# compute the corresponding descriptors ...
D = Main.eval("SHIPsDescriptors.descriptors(desc, at)")
print(D)
Nx = Main.length(Main.desc)
Nat = Main.at.positions.shape[0]
print("Nx = ", Nx, "; Nat = ", Nat, "; D.shape = ", D.shape)
