#
# compares the dumux solution to the analytical solution (Figure 2abc Vanderborght et al 2005)
#
# D. Leitner, 2018
#

import os
import matplotlib.pyplot as plt
from analytic_4abc import *
from vtk_tools import *
import van_genuchten as vg

# manually set absolute path
path = "/home/daniel/workspace/DUMUX/dumux-rosi/build-cmake/rosi_benchmarking/richards1d/"

# run dumux 
os.chdir( path )
os.system( "./richards1d input/b3a.input")
os.system( "./richards1d input/b3b.input")
os.system( "./richards1d input/b3c.input")

# result dumux jan3a (Figure 4a)
for i in range(0,3):
    s_, p_ = read1D_vtp_data(path+"benchmark3a-0000"+str(i+1)+".vtp", False)
    z_ = np.linspace(0,-200,len(s_))
    h_ = vg.pa2head(p_) 
    theta_ = vg.water_content(h_, sand)
    ax1.plot(theta_,z_, "r+")   
  
# result dumux jan3b (Figure 4b)
for i in range(0,3):
    s_, p_ = read1D_vtp_data(path+"benchmark3b-0000"+str(i+1)+".vtp", False)
    z_ = np.linspace(0,-200,len(s_))
    h_ = vg.pa2head(p_) 
    theta_ = vg.water_content(h_, loam)
    ax2.plot(theta_,z_, "r+")
 
# # result dumux jan3c (Figure 4c)
for i in range(0,3):
    s_, p_ = read1D_vtp_data(path+"benchmark3c-0000"+str(i+1)+".vtp", False)
    z_ = np.linspace(0,-200,len(s_))
    h_ = vg.pa2head(p_) 
    theta_ = vg.water_content(h_, clay)
    ax3.plot(theta_,z_, "r+")

plt.show()

