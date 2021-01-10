# Script to generate plots for the 
# second-order topological insulator (Schindler et al.)

# The goal is to make this all-inclusive in order for it to 
# be run on a Compute Canada cluster

import numpy as np
import scipy.sparse as ss
import scipy.sparse.linalg as ssl

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
def unit_block_slab(p, q, nu, zu, t=-1, M = 2.3, D1 = 0.8, D2 = 0.5):
    """
    Block to go into soti_block diagonals (fixed x, varying y)
    """
    Phi = p/q
    spin_nos = 4 # will be 4 unless physics changes
    
    # make diagonals
    cos_diags = [np.cos(2*np.pi*(Phi)*m - zu) for m in range(q)]
    sin_diags = [np.sin(2*np.pi*(Phi)*m - zu) for m in range(q)]
    diags_y = t * np.kron(np.diag(cos_diags),pms.s0_tz()) + D1 * np.kron(np.diag(sin_diags),pms.sz_tx())
    mass_const = M * pms.s0_tz()
    diags_const = np.kron(np.eye(N=q),mass_const)
    A_diags = diags_const + diags_y
    
    # off diagonals y -> y+1 & h.c.
    hop_y = 1/2 * (t * pms.s0_tz() + 1j * D1 * pms.sy_tx() - D2 * pms.s0_ty())
    hop_y_dag = hop_y.conj().T
    
    # put into off diagonals
    A_top_diag = np.kron(np.diag(np.ones(q-1),k=1),hop_y)
    A_bot_diag = np.kron(np.diag(np.ones(q-1), k=-1),hop_y_dag)
    
    A_off_diags = A_top_diag + A_bot_diag

    # corner boundary conditions
    # this is what differentiates the slab
    # from the hinge modes
    A_off_diags[0:spin_nos,spin_nos*(q-1):spin_nos*q] = np.exp(1j*nu)*hop_y_dag
    A_off_diags[spin_nos*(q-1):spin_nos*q,0:spin_nos] = np.exp(-1j*nu)*hop_y
    
    A = A_diags + A_off_diags
    
    return A

# soti block
def soti_block_slab(size, p , q, nu, zu, t = -1, M = 2.3, D1 = 0.8, D2 = 0.5):
    """
    Block to go into H_SOTI diagonals (fixed y, varying x)
    """
    # put unit_blocks into diag
    
    # make blocks array with dims (size,4q,4q)
    blocks = np.zeros((size,4*q,4*q),dtype=complex) 
    
    # fill up
    #xs = linspace(0,size,num=size) # for completeness
    for i in range(size):
        #x = xs[i] # doesn't actually do anything
        blocks[i,:,:] = unit_block_slab(p=p,q=q,nu=nu,zu=zu,t=t,M=M,D1=D1,D2=D2)
        
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

# go for spectra
def kz_spectrum_slab(p,q,nu=0,kz_res=100,ucsize=1,t=-1,M=2.3,D1=0.8,D2=0.5):
    """
    Energies as a function of k_z
    """
    # kz
    kzs = np.linspace(-np.pi,np.pi,num=kz_res,endpoint=False) # to get pretty mid point

    # for each kz, get Es from soti_block
    kz_ret = []
    Es = []

    # set the number of eigs returned
    newsize = q * ucsize
    num_eigs = int((4*newsize**2)/8) # <- let's try this for now and tweak if need be
    
    for kz in kzs:
        H_kz = soti_block_slab(newsize,p,q,nu=nu,zu=kz,t=t,M=M,D1=D1,D2=D2)
        E_kz = ssl.eigsh(H_kz,k=num_eigs,sigma=0,return_eigenvectors=False)
        Es.extend(E_kz)
        kz_ret.extend([kz]*num_eigs)

    return kz_ret, Es

def ky_spectrum_slab(p,q,zu=0,ky_res=100,ucsize=1,t=-1,M=2.3,D1=0.8,D2=0.5):
    """
    Energies as a function of k_z
    """
    # kz
    kys = np.linspace(-np.pi,np.pi,num=ky_res,endpoint=False) # to get pretty mid point

    # for each kz, get Es from soti_block
    ky_ret = []
    Es = []

    # set the number of eigs returned
    newsize = q * ucsize
    num_eigs = int((4*newsize**2)/8) # <- let's try this for now and tweak if need be
    
    for ky in kys:
        H_ky = soti_block_slab(newsize,p,q,nu=ky,zu=zu,t=t,M=M,D1=D1,D2=D2)
        E_ky = ssl.eigsh(H_ky,k=num_eigs,sigma=0,return_eigenvectors=False)
        Es.extend(E_ky)
        ky_ret.extend([ky]*num_eigs)

    return ky_ret, Es

# butterfly stuff
def get_phis_eps(qmax=10,ucsize=1,nu=0,zu=0,t=-1,M=2.3,D1=0.8,D2=0.5):
    """
    Phis and Energies required to plot Hofstadter's butterfly. Only for 
    a given kz.
    """
    # initialize
    phi = []
    eps = []

    # fill up
    for q in range(1,qmax):
        newsize = q*ucsize
        for p in range(0,q):
            # side length of the square H
            H_dim = 4*newsize**2
            # add phi
            phi.extend([p/q]*H_dim + [(q-p)/q]*H_dim)
            # get H and eps
            H_pq = soti_block_slab(size=newsize,p=p,q=q,nu=nu,zu=zu,t=t,M=M,D1=D1,D2=D2)
            eigs_pq = ssl.eigsh(H_pq,k=H_dim,return_eigenvectors=False)
            eps.extend([eigs_pq]*2)
    
    # convert into ndarray
    phi = np.asarray(phi)
    eps = np.concatenate(eps)
    eps = np.asarray(eps)

    return phi, eps


# import Fraction
from fractions import Fraction

def get_phis_eps_spaced(Phi,q=20,n=10,ucsize=1,nu=0,zu=0,t=-1,M=2.3,D1=0.8,D2=0.5):
    """
    Inspect a specific Phi region in evenly spaced intervals.
    """
    # make evenly spaced phis
    phis=np.linspace(Phi-n*1/q,Phi+n*1/q,num=2*n+1,endpoint=True)

    # how many eigenvalues will it return?
    k=int(4*q**2/16)

    # set phi and E arrays to return
    phi_ret=np.zeros(k*len(phis),dtype=float)
    eps_ret=np.zeros(k*len(phis),dtype=float)

    # fill them up
    for i in range(len(phis)):
        phi=phis[i]
        phi=Fraction(phi).limit_denominator()
        p=phi.numerator
        q=phi.denominator
        size=q*ucsize
        H_phi=soti_block_slab(size=size,p=p,q=q,nu=nu,zu=zu,t=-1,M=2.3,D1=0.8,D2=0.5)
        E_phi=ssl.eigsh(H_phi,k=k,sigma=0,return_eigenvectors=False)
        eps_ret[i*k:(i+1)*k]=E_phi
        phi_ret[i*k:(i+1)*k]=np.full(k,phi)

    return phi_ret, eps_ret

ks=np.linspace(-np.pi,np.pi,num=101,endpoint=True)

# main scripts
def main_butterfly(qmax=50,nu=0,ucsize=1,t=-1,M=2.3,D1=0.8,D2=0.5):
    import sys
    # get kz from argv
    args = sys.argv
    kz_idx = int(args[1])
    kz = ks[kz_idx]
    # run program
    phi, eps = get_phis_eps(qmax=qmax,nu=nu,zu=kz,ucsize=ucsize,t=t,M=M,D1=D1,D2=D2)
    # save in file
    with open("soti_slab_butterfly_data.csv","a") as f:
        np.savetxt(f,(phi,eps),delimiter=',')

def main_ky_spectra(p,q,ky_res=100,ucsize=1,t=-1,M=2.3,D1=0.8,D2=0.5):
    import sys
    # get kz from argv
    args = sys.argv
    kz_idx = int(args[1])
    kz = ks[kz_idx]
    # run program
    ky,E=ky_spectrum_slab(p=p,q=q,zu=kz,ky_res=ky_res,t=t,M=M,D1=D1,D2=D2)
    # save in file
    with open("soti_slab_ky_spectrum_data.csv","a") as f:
        np.savetxt(f,(ky,E),delimiter=',')

def main_kz_spectra(p,q,kz_res=100,ucsize=1,t=-1,M=2.3,D1=0.8,D2=0.5):
    import sys
    # get ky from argv
    args = sys.argv
    ky_idx = int(args[1])
    ky = ks[ky_idx]
    # run program
    kz,E=kz_spectrum_slab(p=p,q=q,nu=ky,kz_res=kz_res,t=t,M=M,D1=D1,D2=D2)
    # save in file
    with open("soti_slab_kz_spectrum_data.csv","a") as f:
        np.savetxt(f,(kz,E),delimiter=',')

# for parallel code in kz
def p_main_kz_spectra(p,q,kz_res=100,ucsize=1,t=-1,M=2.3,D1=0.8,D2=0.5):
    import sys
    # get kz from argv
    args = sys.argv
    kz_idx = int(args[1])
    kz = ks[kz_idx]
    # run program
    H = soti_block_slab(size=q)
    # save in file
    with open("soti_slab_kz_spectrum_data.csv","a") as f:
        np.savetxt(f,(kz,E),delimiter=',')

if __name__ == "__main__":
    #main_butterfly(qmax=50,nu=0,ucsize=1,t=-1,M=2.3,D1=0.8,D2=0.5)
    #main_ky_spectra(p=1,q=10,ky_res=100)
    main_kz_spectra(p=1,q=10,kz_res=1000)
