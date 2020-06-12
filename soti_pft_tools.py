# Tools for partial Fourier transform of the SOTI

from numpy import *
from soti_tools import *
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
    
    M = M_diags + M_off_diags
    
    return M

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