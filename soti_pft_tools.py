# Tools for partial Fourier transform of the SOTI

from numpy import *
from soti_tools import * # maybe not a good idea to cross
                        # contaminate
from scipy.linalg import block_diag
import scipy.sparse as ss
import scipy.sparse.linalg as ssl

# define kroneckers of pauli matrices
s0_ty = kron(sig(0),sig(2))
s0_tz = kron(sig(0),sig(3))
sx_tx = kron(sig(1),sig(1))
sy_tx = kron(sig(2),sig(1))
sz_tx = kron(sig(3),sig(1))

def unit_block(size, p, q, zu, t=1, M = 2.3, D1 = 0.8, D2 = 0.5):
    """
    Block to go into soti_block diagonals
    """
    Phi = p/q
    
    # make diagonals
    cos_diags = [cos(2*pi*(Phi)*y - zu) for y in range(size)]
    sin_diags = [sin(2*pi*(Phi)*y - zu) for y in range(size)]
    diags_y = t * kron(diag(cos_diags),s0_tz) + D1 * kron(diag(sin_diags),sz_tx)
    mass_const = M * s0_tz
    diags_const = kron(eye(N=size),mass_const)
    A_diags = diags_const + diags_y
    
    # off diagonals y -> y+1 & h.c.
    hop_y = 1/2 * (t * s0_tz + 1j * D1 * sy_tx - D2 * s0_ty)
    hop_y_dag = hop_y.conj().T
    
    # put into off diagonals
    A_top_diag = kron(diag(ones(size-1),k=1),hop_y)
    A_bot_diag = kron(diag(ones(size-1), k=-1),hop_y_dag)
    
    A_off_diags = A_top_diag + A_bot_diag
    
    A = A_diags + A_off_diags
    
    return A

def soti_block(size, p , q, zu, t = 1, M = 2.3, D1 = 0.8, D2 = 0.5):
    """
    Block to go into H_SOTI diagonals
    """
    # put unit_blocks into diag
    
    # make blocks array with dims (size,4size,4size)
    blocks = zeros((size,4*size,4*size),dtype=complex) 
    
    # fill up
    xs = linspace(0,size,num=size) # for completeness
    for i in range(size):
        x = xs[i]
        blocks[i,:,:] = unit_block(size=size,p=p,q=q,zu=zu,t=t,
                                   M=M,D1=D1,D2=D2)
        
    # put in diagonal
    M_diags = ss.block_diag(blocks).toarray()
    
    # off diagonals x -> x+1 & h.c.
    hop_x = 1/2 * (t * s0_tz + 1j * D1 * sx_tx + D2 * s0_ty)
    hop_x_dag = hop_x.conj().T
    
    # fill up to identity
    hop_x_mat = kron(eye(N=size), hop_x)
    hop_x_mat_dag = kron(eye(N=size), hop_x_dag)
    
    # put these "identity" matrices on the off-diagonals
    ### double check the math for this section please
    M_top_diag = kron(diag(ones(size-1), k=1), hop_x_mat)
    M_bot_diag = kron(diag(ones(size-1), k=-1), hop_x_mat_dag)
    
    M_off_diags = M_top_diag + M_bot_diag
    
    MAT = M_diags + M_off_diags
    
    return MAT

def H_SOTI(size, p, q, t = 1, M = 2.3, D1 = 0.8, D2 = 0.5):
    """
    SOTI Hamiltonian Fourier transformed in z and real in x,y
    """
    # q == size
    if int(size) != int(q):
        raise Exception("size must be the same as q")
        return

    # put blocks in diagonal - 1 for every zu
    blocks = zeros((size,4*size**2,4*size**2),dtype=complex)
    zus = linspace(-pi,pi,size,endpoint = True)
    for i in range(size):
        zu = zus[i]
        blocks[i,:,:] = soti_block(size=size,p=p,q=q,zu=zu,t=t,
                                   M=M,D1=D1,D2=D2)
        
    # use sparse.block_diag instead of scipy.ditto because it takes in an array
    H = ss.block_diag(blocks).toarray() # <- still needs testing
    
    return H

def kz_spectrum(size,p,q,kz_res=10,t=1,M=2.3,D1=0.8,D2=0.5):
    """
    Energies as a function of k_z
    """
    # q == size
    if int(size) != int(q):
        raise Exception("size must be the same as q")
        return

    # kz
    kz = linspace(pi-1,pi+1,kz_res,endpoint=True)

    # for each kz, get Es from soti_block
    kz_ret = []
    Es = []

    for k in kz:
        H_kz = soti_block(size,p,q,zu=k,t=t,M=M,D1=D1,D2=D2)
        H_kz_dim = int(4*size**2)
        E_kz = ssl.eigsh(H_kz,k=H_kz_dim,return_eigenvectors=False)
        Es.extend(E_kz)
        kz_ret.extend([k]*H_kz_dim)

    return kz_ret, Es

def get_phis_eps(qmax=10,zu=0,t=1,M=2.3,D1=0.8,D2=0.5):
    """
    Phis and Energies required to plot Hofstadter's butterfly. Only for 
    a given kz.
    """
    # initialize
    phi = []
    eps = []
    eps_test = []

    # fill up
    for q in range(1,qmax):
        for p in range(0,q):
            # side length of the square H
            H_dim = 4*q**2
            # add phi
            phi.extend([p/q]*H_dim + [(q-p)/q]*H_dim)
            # get H and eps
            H_pq = soti_block(size=q,p=p,q=q,zu=zu,t=t,M=M,D1=D1,D2=D2)
            eigs_pq = ssl.eigsh(H_pq,k=H_dim,return_eigenvectors=False)
            eps.extend([eigs_pq]*2)
    
    # convert into ndarray
    phi = asarray(phi)
    eps = concatenate(eps)
    eps = asarray(eps)

    return phi, eps

def sum_over_kz(qmax=10,kz_res=10,ucs=1,t=1,M=2.3,D1=0.8,D2=0.5):
    """
    Sum over plane wave direction (z) to get correct butterfly pattern
    """
    # kz
    kz = linspace(-pi/ucs,pi/ucs,kz_res,endpoint=True)

    # initialize
    phi = []
    eps = []

    # fill up
    for k in kz:
        phi_k, eps_k = get_phis_eps(qmax=qmax,zu=k,t=t,M=M,D1=D1,D2=D2)
        phi.extend(phi_k)
        eps.extend(eps_k)

    # convert into ndarray
    phi = asarray(phi)
    eps = asarray(eps)

    return phi, eps 