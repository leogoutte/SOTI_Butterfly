# Script to inspect Hofstadter's butterfly for the 
# second-order topological insulator (Schindler et al.)
# near phi = 0

# The goal is to make this all-inclusive in order for it to 
# be run on a Compute Canada cluster

import numpy as np
import scipy.sparse as ss
import scipy.sparse.linalg as ssl
import sys

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
    hop_x = 1/2 * (t * pms.s0_tz() + 1j * D1 * pms.sx_tx() + D2 * pms.s0_ty())
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

def inspect_butterfly(pmax,qmax,zu,ucsize=1,t=-1,M=2.3,D1=0.8,D2=0.5):
    """
    Zooms in on low Phi and near-fermi level energies for Schindler's SOTI butterfly.
    Contrary to other butterfly plot programs, the Phis here are evenly spaced. 
    """
    # make phis
    phi = []
    eps = []

    # fill up
    for q in range(1,qmax+1):
        newsize = q * ucsize
        num_eigs = int(2*q) # sweet spot
        for p in range(pmax+1):
            H_pq = soti_block(size=newsize,p=p,q=q,zu=zu,t=t,M=M,D1=D1,D2=D2)
            eps_pq = ssl.eigsh(H_pq,k=num_eigs,sigma=0,return_eigenvectors=False)
            phi.extend([p/q]*num_eigs) # no need to get (q-p)/q
            eps.extend(eps_pq)

    return phi, eps

# set up kzs
kzs = np.linspace(-np.pi,np.pi,num=101,endpoint=True)

if __name__ == "__main__":
    # get kz from argv
    args = sys.argv
    kz_idx = int(args[1])
    kz = kzs[kz_idx]
    # run program
    phi, eps = inspect_butterfly(pmax=1,qmax=50,zu=kz,ucsize=1,t=-1,M=2.3,D1=0.8,D2=0.5)
    # save in file
    with open("soti_inspect_data.csv","a") as f:
        np.savetxt(f,(phi,eps),delimiter=',')

### LPBG
