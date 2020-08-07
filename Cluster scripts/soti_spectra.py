# Script to generate Energy vs. kz spectra for the 
# second-order topological insulator (Schindler et al.)

# The goal is to make this all-inclusive in order for it to 
# be run on a Compute Canada cluster

import numpy as np
import scipy.sparse as ss
import scipy.sparse.linalg as ssl
import matplotlib.pyplot as plt

# pauli matrices
def sig(n):
    # pauli matrices
    # n = 0 is identity, n = 1,2,3 is x,y,z resp.
    if n == 0:
        a = np.identity(2, dtype = complex)
    if n == 1:
        a = np.array([[0 , 1],[1 , 0]], dtype = complex)
    if n == 2:
        a = np.array([[0 , -1j],[1j , 0]], dtype = complex)
    if n == 3:
        a = np.array([[1 , 0],[0 , -1]], dtype = complex)
    return a

class pauli:
    def __init__(self,s0,sx,sy,sz):
        self.s0 = s0
        self.sx = sx
        self.sy = sy
        self.sz = sz
    def s0_ty(self):
        return np.kron(self.s0,self.sy)
    def s0_tz(self):
        return np.kron(self.s0,self.sz)
    def sx_tx(self):
        return np.kron(self.sx,self.sx)
    def sy_tx(self):
        return np.kron(self.sy,self.sx)
    def sz_tx(self):
        return np.kron(self.sz,self.sx)

pms = pauli(sig(0),sig(1),sig(2),sig(3))


# unit block
def unit_block(size, p, q, zu, t=-1, M = 2.3, D1 = 0.8, D2 = 0.5):
    """
    Block to go into soti_block diagonals (fixed x, varying y)
    """
    Phi = p/q
    
    # make diagonals
    cos_diags = [np.cos(2*np.pi*(Phi)*y - zu) for y in range(size)]
    sin_diags = [np.sin(2*np.pi*(Phi)*y - zu) for y in range(size)]
    diags_y = t * np.kron(np.diag(cos_diags),pms.s0_tz()) + D1 * np.kron(np.diag(sin_diags),pms.sz_tx())
    mass_const = M * pms.s0_tz()
    diags_const = np.kron(np.eye(N=size),mass_const)
    A_diags = diags_const + diags_y
    
    # off diagonals y -> y+1 & h.c.
    hop_y = 1/2 * (t * pms.s0_tz() + 1j * D1 * pms.sy_tx() - D2 * pms.s0_ty())
    hop_y_dag = hop_y.conj().T
    
    # put into off diagonals
    A_top_diag = np.kron(np.diag(np.ones(size-1),k=1),hop_y)
    A_bot_diag = np.kron(np.diag(np.ones(size-1), k=-1),hop_y_dag)
    
    A_off_diags = A_top_diag + A_bot_diag
    
    A = A_diags + A_off_diags
    
    return A

# soti block
def soti_block(size, p , q, zu, t = -1, M = 2.3, D1 = 0.8, D2 = 0.5):
    """
    Block to go into H_SOTI diagonals (fixed y, varying x)
    """
    # put unit_blocks into diag
    
    # make blocks array with dims (size,4size,4size)
    blocks = np.zeros((size,4*size,4*size),dtype=complex) 
    
    # fill up
    #xs = linspace(0,size,num=size) # for completeness
    for i in range(size):
        #x = xs[i] # doesn't actually do anything
        blocks[i,:,:] = unit_block(size=size,p=p,q=q,zu=zu,t=t,
                                   M=M,D1=D1,D2=D2)
        
    # put in diagonal
    M_diags = ss.block_diag(blocks)
    
    # off diagonals x -> x+1 & h.c.
    hop_x=1/2*(t*pms.s0_tz()+1j*D1*pms.sx_tx()+D2*pms.s0_ty())
    hop_x_dag = hop_x.conj().T
    
    # fill up to identity
    hop_x_mat = np.kron(np.eye(N=size), hop_x)
    hop_x_mat_dag = np.kron(np.eye(N=size), hop_x_dag)
    
    # put these "identity" matrices on the off-diagonals
    ### double check the math for this section please
    M_top_diag = np.kron(np.diag(np.ones(size-1), k=1), hop_x_mat)
    M_bot_diag = np.kron(np.diag(np.ones(size-1), k=-1), hop_x_mat_dag)
    
    M_off_diags = M_top_diag + M_bot_diag
    
    MAT = M_diags + M_off_diags
    
    return MAT

def H_SOTI(size, p, q, t = -1, M = 2.3, D1 = 0.8, D2 = 0.5):
    """
    SOTI Hamiltonian Fourier transformed in z and real in x,y
    """

    # put blocks in diagonal - 1 for every zu
    blocks = np.zeros((size,4*size**2,4*size**2),dtype=complex)
    zus = np.linspace(-np.pi,np.pi,size,endpoint = True)
    for i in range(size):
        zu = zus[i]
        blocks[i,:,:] = soti_block(size=size,p=p,q=q,zu=zu,t=t,
                                   M=M,D1=D1,D2=D2)
        
    # use sparse.block_diag instead of scipy.ditto because it takes in an array
    H = ss.block_diag(blocks).toarray() # <- still needs testing
    
    return H

def kz_spectrum(p,q,kz_res=10,ucsize=1,t=-1,M=2.3,D1=0.8,D2=0.5):
    """
    Energies as a function of k_z.
    """
    # kz
    kzs = np.linspace(-np.pi,np.pi,num=kz_res+1,endpoint=True) # to get pretty mid point

    # for each kz, get Es from soti_block
    kz_ret = []
    Es = []

    # set the number of eigs returned
    newsize = q * ucsize
    num_eigs = int((4*newsize**2)) # <- let's try this for now and tweak if need be
    
    for kz in kzs:
        H_kz = soti_block(newsize,p,q,zu=kz,t=t,M=M,D1=D1,D2=D2)
        E_kz = ssl.eigsh(H_kz,k=num_eigs,sigma=0,return_eigenvectors=False)
        Es.extend(E_kz)
        kz_ret.extend([kz]*num_eigs)

    return kz_ret, Es

def spectrum_plots_kz(ps=[0,1,10],qs=[20,20,20],kz_res=10,ucsize=1,t=-1,M=2.3,D1=0.8,D2=0.5):
    """
    Plots of Energy as a function of k for the SOTI for various magnetic flux 
    """
    # initialize
    all_ks=[]
    all_Eks=[]

    # fill them up
    for i in range(len(ps)):
        p = ps[i]
        q = qs[i]
        ks, Eks = kz_spectrum(p=p,q=q,kz_res=kz_res,ucsize=ucsize,t=t,M=M,D1=D1,D2=D2)

        all_ks.append(ks)
        all_Eks.append(Eks)

    return all_ks, all_Eks

# run it
if __name__ == "__main__":
    import csv
    all_ks, all_Es = spectrum_plots_kz(ps=[2,3,4],qs=[20,20,20],kz_res=100,ucsize=3)
    # save arrays to files
    # if list with different lengths
    with open("all_ks_soti.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(all_ks)  
    with open("all_Es_soti.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(all_Es) 
    # if np array  
    #savetxt("all_ks_soti.csv",all_ks,delimiter=',')
    #savetxt("all_Es_soti.csv",all_Es,delimiter=',')
    

### LPBG