# Benchmark 2 (in 3D)
#
# compares the dumux solution to the analytical solution (Figure 3, Vanderborght et al 2005)
#
# D. Leitner, 2018
#
import os
import matplotlib.pyplot as plt
from vtk_tools import *
import van_genuchten as vg
import analytic_b2

# go to the right place
path = os.path.dirname(os.path.realpath(__file__))
os.chdir(path)
os.chdir("../../../build-cmake/rosi_benchmarking/soil")

# run dumux
np_ = 1  # number of processors
if np_ == 1:
    os.system("./richards3d benchmarks_3d/b2.input")
else:
    os.system("mpirun -n " + str(np_) + " ./richards3d benchmarks_3d/b2.input -Grid.Overlap 0")

# result dumux jan1 (Figure 2a)
s_, p_, z_ = read3D_vtp("benchmark3d_2-00001", np_)
h_ = vg.pa2head(p_)
plt.plot(h_, z_ * 100 - 53, "r+")

np.savetxt("dumux3d_b2", np.vstack((z_ - .53, h_)), delimiter = ",")

plt.show()
