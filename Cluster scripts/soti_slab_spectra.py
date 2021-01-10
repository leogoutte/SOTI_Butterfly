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

def get_indices(waves):
    """
    Returns surface and bulk indices for a given set of waves
    """
    prob_ = np.abs(waves)**2
    # batch
    prob = [np.sum(prob_[i:i+4,:], axis=0) for i in range(0, len(waves[:,0]), 4)]
    prob = np.asarray(prob)
    prob_tot = np.sum(prob, axis=0)
    
    # cutoff
    length = np.size(prob[:,0])
    len10 = int(length/10)
    flags = np.zeros((prob.shape[1]), dtype=int)
    # hinges
    # 50% within 10% of corners

    # surface
    # 50% within 10% of surfaces
    # not already labelled hinges
    prob_left = np.sum(prob[0:len10,:], axis=0)
    frac_left = prob_left/prob_tot

    prob_right = np.sum(prob[length-len10:length,:], axis=0)
    frac_right = np.divide(prob_right, prob_tot)

    for i in range(len(flags)):
        if frac_left[i]>0.5 or frac_right[i]>0.5:
            flags[i] = 1
            
    indices = [i for i, x in enumerate(flags) if x == 1]
    indices0 = [i for i, x in enumerate(flags) if x == 0]
    
    return indices, indices0

def kz_spectrum_single(kz,p,q,nu=0,ucsize=1,t=-1,M=2.3,D1=0.8,D2=0.5):
    newsize = q * ucsize
    num_eigs = int((4*newsize)) # <- let's try this for now and tweak if need be

    H_kz = soti_block_slab(newsize,p,q,nu=nu,zu=kz,t=t,M=M,D1=D1,D2=D2)
    E_kz,waves = ssl.eigsh(H_kz,k=num_eigs,sigma=0,return_eigenvectors=True)
    idx,idx0=get_indices(waves)
    kz_surf = np.full(len(idx),kz)
    kz_bulk = np.full(len(idx0),kz)
    Es_surf = E_kz[idx]
    Es_bulk = E_kz[idx0]

    return kz_surf, Es_surf, kz_bulk, Es_bulk

# run it
if __name__ == "__main__":
    import sys
    kzs=np.linspace(-np.pi,np.pi,num=101,endpoint=True)
    # get kz from argv
    args = sys.argv
    kz_idx = int(args[1])
    kz = kzs[kz_idx]
    # run program
    kz_surf,Es_surf,kz_bulk,Es_bulk=kz_spectrum_single(kz,p=1,q=50,nu=0,ucsize=1,t=-1,M=2.3,D1=0.8,D2=0.5)
    # np array  
    with open("soti_slab_spectra_data_surf.csv","a") as f:
        np.savetxt(f,(kz_surf,Es_surf),delimiter=',')
    with open("soti_slab_spectra_data_bulk.csv","a") as f:
        np.savetxt(f,(kz_bulk,Es_bulk),delimiter=',')
    
### LPBG