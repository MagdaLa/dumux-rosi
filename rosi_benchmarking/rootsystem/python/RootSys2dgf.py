import numpy as np
import math
import sys
import os
import re
from io import StringIO  # StringIO behaves like a file object
'''
Creates dgf files for a defined root system
'''

def createDGF_1Droots(filename, nodes, seg, params = np.zeros((0, 0))):
    file = open(filename, "w")  # write file

    nop = params.shape[0]  # number of parameters
    file.write("DGF\n")
    file.write('Vertex\n')
    file.write('parameters 0\n')


    add = np.array([[0, 0, 0]],dtype=np.float)
    nodes_ = np.concatenate((add, nodes[:,0:3]), axis=0)
    for i in range(0, len(nodes_)):
        file.write('{:g} {:g} {:g} \n'.format(nodes_[i, 0], nodes_[i, 1], nodes_[i, 2]))

    file.write('#\n');
    file.write('Simplex\n');
    if nop > 0:
        file.write('parameters {:d}\n'.format(nop))
    for i in range(0, len(seg)):
        file.write('{:g} {:g}'.format(seg[i, 0], seg[i, 1]))
        for j in range(0, nop):
            file.write(' {:g}'.format(params[j, i]))
        file.write(' \n')

    # not used...
    file.write('#\nBOUNDARYSEGMENTS\n2 0\n')
    file.write('3 {:g}\n'.format(len(seg)))
    file.write('#\nBOUNDARYDOMAIN\ndefault 1\n#\n')
    file.close()


specnme = "IV_Soil_3D"               ##SPECIFY ROOTSYS NAME
dir = ("RootSys/"+specnme)     
RS_sgn = []
RS_age = []
filenum = len(os.listdir(dir))       #number of RootSys files
for fname in os.listdir(dir):
    if fname.endswith("_RootSys"): 
        RS_age_ = ([float(s) for s in re.findall(r'-?\d+\.?\d*', fname)])
        with open(dir+"/"+fname, "r") as f:
            content = f.readlines()
            RS_sgn_ = float(content[22].split(',')[0])
        RS_age.append(RS_age_)      #age and segment number of all files 
        RS_sgn.append(RS_sgn_)
RS_age.sort()
RS_sgn.sort()


sgn=[]
fname = "DAP"+(str(int(np.asarray(RS_age[-1]))))+"_RootSys"
with open(dir+"/"+fname, "r") as f:        #last file
    content = f.readlines()
    sgn = float(content[22].split(',')[0])

# read relevant data
table1 = []  # holds: segID#    x          y          z      prev or  br#  length   surface  mass
table2 = []  # holds: origination time

x=[]
carray1 = []
clist1 = []
carray2 = []
clist2 = []
i = 26  # start row
while i < (26+sgn*2):
    line = content[i]
    table1 = [float(x) for x in content[i].split()]    #read in floats
    table2 = [float(x) for x in content[i+1].split()]
    clist1.append(table1)                              #append each line to a list
    clist2.append(table2)  
    i += 2

carray1 = np.asarray(clist1)  #convert the list to an array 
carray2 = np.asarray(clist2) 
nnz = len(carray1)

ID = (carray1[:,[0]])
nodes = (carray1[:,[1,2,3]])* 1.e-1       #convert from mm to cm  
if (nodes[1,2]>=0): 
    nodes[:,2] = -1*nodes[:,2]
prev = (carray1[:,[4]])
seg = np.concatenate((carray1[:,[4]], carray1[:,[0]]), axis=1)
age = (carray2[:,[0]])
surf = (carray1[:,[8]])
a = (carray1[:,[8]] / (carray1[:,[7]] * 2 * math.pi))* 1.e-1 #convert from mm to cm   
brn =  (carray1[:,[6]])
leng =  (carray1[:,[7]])* 1.e-1 #convert from mm to cm   
mxbrn = int(max(brn))

##the following values were taken from Leitner 
#kr_ = np.ones((1, nnz)) * 5.e-6 * 10.e-6 / (100*24*3600)    #cm³ hPa-1 d-1 converted to SI unit m³ Pa-1 s-1
#kz_ = np.ones((1, nnz)) * 5.e-4 * 10.e-2 / (100*24*3600)    #cm³ hPa-1 d-1 converted to SI unit m Pa-1 s-1

kz_ = np.ones((1, nnz))*5.184e+6
kr_ = np.ones((1, nnz))*0.00499963




# make an array with seg ID of first node of every branch + branch number + previous 
fiID = np.zeros(mxbrn)
fiprev = np.zeros(mxbrn)
prebr = np.zeros(mxbrn)
for i in range(1,mxbrn+1):
    res = next(x for x, val in enumerate(brn) if val == i)     #index of first segment of branch i
    fiID[i-1]= ID[res]                                         #IDs of first branch segements
    fiprev[i-1] = prev[res]                                    #IDs of previous segemnts of first branch segements
    prebr[i-1] = brn[int(fiprev[i-1])]                         #number of previous branch 
prebr[0] = 0                                                   #taproot system: previous branch of branch 1 has number 0    

fibr = np.arange(1,mxbrn+1,1)          #list of unique branches 
typ = np.zeros(mxbrn)             #preallocate branch type array 

#find branch types
idx = (np.where(prebr==0))             #index of taproot branch
typ[idx[0]] = 1                       #give tproot branch order 1
for i in range(1,10):                  #consider maximum 10 differnt branch types 
    bridx = (np.where(typ==i))        #index of branches with type i
    dumpre = fibr[bridx[0]]            #numbers of branches that have type i 
    idxpre = (np.in1d(prebr,dumpre))   #index of branches that have a prebranch with type i 
    typ[idxpre]=(i+1)                 #type of branches with prebranch type i = i+1


#distribute types on all segments of each branch 
typs = np.empty_like(leng)
for i in range(1,mxbrn+1):
    idx = (np.where(fibr==i))          #find index of branch i
    tt = typ[idx[0]]                  #type of branch i
    ix = (np.where(brn==i))            #index of all branches i 
    typs[ix[0]] = tt


#iterate through Root Sys files and compute age of each segement
lengacc = leng                         #lengacc is overwritten, if it is not leng
RS_sgn = np.array(RS_sgn, dtype=np.int)
for i in range(0,filenum):
   IDdum = ID[:RS_sgn[i]]              # IDs of branches of current file 
   brdum = brn[:RS_sgn[i]]             # branch numbers of branches of current file 
   typdum = typs[:RS_sgn[i]]          # types of branches of current file 
   prevdum = prev[:RS_sgn[i]]          #previous segments branches of current file 
     

   typlist = np.unique(typdum)               #unique types of branches in file i
   brlist = np.unique(brdum)                 #unique branch numbers in file i
   
   for j in (typlist):  
      for k in (brlist): 
         idx_ = np.where((typdum==j) & (brdum == k))
         if i == 0: 
            idx = idx_[0]
         else: 
            previx = np.arange(0,RS_sgn[i-1]-1,1)#array of segemnt indices of previous file 
            xx = np.in1d(idx_, previx, invert=True)   #remove those elements whose age has already been determined in the previous file 
            ln = len(idx_[0])              #adjust to array size
            idx = idx_[0][xx[:ln]]
      
         brtyp = brdum[idx]                #branch numbers of type j and branch k
         IDtyp = IDdum[idx]                #IDs of branches of type j and branch k
     
         
         if len(idx):
            #lengacc[idx[0][0]]= leng[idx[0][0]]    #acc length of segment 1 of branch k, type j 
            if len(idx)>1:
               line = idx[1:]
               for m in (line):                         #accumulate length for branch k, type j
                  lengacc[m] = lengacc[m-1]+leng[m]
            
            dummy = prev[idx[0]]
            age[idx[0]] = age[dummy.astype(int)]      # age of previous segment of first segment of branch k, type j        
            age[max(idx)] = np.asarray(RS_age[i])     # age of last segment of branch k 
            dage = age[max(idx)]- age[idx[0]]                 #age difference between firsat and last segemtn of branch k, type j 
            dlen = lengacc[max(idx)]
            v = dlen/dage                                #cm / d 
            if len(idx)>1:
               line = idx[1:]
               for m in (line):                          #accumulate length for branch k, type j
                  age[m] = lengacc[m]/v+age[idx[0]]  
            


age_ = (max(RS_age)-age)*24*3600
surf = np.array(surf) * 1.e-2          #convert to cm2 
leng = np.array(leng) * 1.e-1          #convert to cm 
a = np.array(a) * 1.e-1                #convert to cm          
order = np.array(typs) - 1;
#age in [d], kz in [cm4 hPa-1 d-1], kr in [cm hPa-1 d-1]
order[0]=-1
brn[0]=-1
#idx = np.where(order>=1)
#order[idx[0]] = 1
nodes = nodes*1.e-2
params = np.vstack((order.T, brn.T, surf.T, leng.T, a.T, kz_, kr_, age_.T))

createDGF_1Droots("../grids/"+specnme+".dgf", nodes, seg, params) 
