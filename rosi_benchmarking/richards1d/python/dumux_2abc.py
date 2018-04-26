#
# compares the dumux solution to the analytical solution (Figure 2abc Vanderborght et al 2005)
#
# D. Leitner, 2018
#

import os
import matplotlib.pyplot as plt
from analytic_2abc import *
from vtk_tools import *
import van_genuchten as vg

# manually set absolute path
path = "/home/daniel/workspace/DUMUX/dumux-rosi/build-cmake/rosi_benchmarking/richards1d/"

# # run dumux 
os.chdir( path )
os.system( "./richards1d input/b1a.input")
os.system( "./richards1d input/b1b.input")
os.system( "./richards1d input/b1c.input")

# result dumux jan1 (Figure 2a)
s_, p_ = read1D_vtp_data(path+"benchmark1a-00001.vtp", False)
z_ = np.linspace(0,-200,len(s_))
h_ = vg.pa2head(p_) 
ax1.plot(h_,z_, "r+")
 
# result dumux jan1 (Figure 2b)
s_, p_ = read1D_vtp_data(path+"benchmark1b-00001.vtp", False)
z_ = np.linspace(0,-200,len(s_))
h_ = vg.pa2head(p_) 
ax2.plot(h_,z_, "r+")
 
# result dumux jan1 (Figure 2c)
s_, p_ = read1D_vtp_data(path+"benchmark1c-00001.vtp", False)
z_ = np.linspace(0,-200,len(s_))
h_ = vg.pa2head(p_) 
ax3.plot(h_,z_, "r+")

plt.show()

