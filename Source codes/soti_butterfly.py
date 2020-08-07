# Script to generate Hofstadter's butterfly for the 
# second-order topological insulator (Schindler et al.)

# The goal is to make this all-inclusive in order for it to 
# be run on a Compute Canada cluster

# modules
from numpy import *
import scipy.sparse as ss
import scipy.sparse.linalg as ssl
import matplotlib.pyplot as plt

# gcd
def gcd(a,b):
    if b == 0:
        return a
    else:
        return gcd(b,a%b)

# pauli matrices
def sig(n):
    # pauli matrices
    # n = 0 is identity, n = 1,2,3 is x,y,z resp.
    if n == 0:
        a = identity(2, dtype = complex)
    if n == 1:
        a = array([[0 , 1],[1 , 0]], dtype = complex)
    if n == 2:
        a = array([[0 , -1j],[1j , 0]], dtype = complex)
    if n == 3:
        a = array([[1 , 0],[0 , -1]], dtype = complex)
    return a

# define kroneckers of pauli matrices
s0_ty = kron(sig(0),sig(2))
s0_tz = kron(sig(0),sig(3))
sx_tx = kron(sig(1),sig(1))
sy_tx = kron(sig(2),sig(1))
sz_tx = kron(sig(3),sig(1))

def unit_block(size,p,q,zu,t=-1,M=2.3,D1=0.8,D2=0.5):
    """
    4Lx x 4Lx block to go into soti_block diagonals
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

def soti_block(size,p,q,zu,t=-1,M=2.3,D1=0.8,D2=0.5):
    """
    4LxLy x 4LxLy matrix whose eigenvalues are the energies
    of the SOTI
    """
    # put unit_blocks into diag
    
    # make blocks array with dims (size,4size,4size)
    blocks = zeros((size,4*size,4*size),dtype=complex) 
    
    # fill up
    xs = linspace(0,size,num=size) # for completeness
    for i in range(size):
        x = xs[i]
        blocks[i,:,:] = unit_block(size=size,p=p,q=q,zu=zu,t=t,M=M,D1=D1,D2=D2)
        
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

def get_phis_eps(qmax=10,ucsize=1,zu=0,t=-1,M=2.3,D1=0.8,D2=0.5):
    """
    Phis and Energies required to plot Hofstadter's butterfly for 
    a given kz
    """
    # initialize
    phi = []
    eps = []

    # phi = 0
    newsize_0 = qmax*ucsize
    H_dim_0 = 4*newsize_0**2
    # add phi
    phi.extend([0]*H_dim_0 + [1]*H_dim_0)
    # get H and eps
    H_pq_0 = soti_block(size=newsize_0,p=0,q=qmax,zu=zu,t=t,M=M,D1=D1,D2=D2)
    eigs_pq_0 = ssl.eigsh(H_pq_0,k=H_dim_0,return_eigenvectors=False)
    eps.extend([eigs_pq_0]*2)

    # fill up rest
    for q in range(1,qmax):
        newsize = q * ucsize
        for p in range(1,q):
            if gcd(p,q) == 1:
                # side length of the square H
                H_dim = 4*newsize**2
                # add phi
                phi.extend([p/q]*H_dim + [(q-p)/q]*H_dim)
                # get H and eps
                H_pq = soti_block(size=newsize,p=p,q=q,zu=zu,t=t,M=M,D1=D1,D2=D2)
                eigs_pq = ssl.eigsh(H_pq,k=H_dim,return_eigenvectors=False)
                eps.extend([eigs_pq]*2)
            print("Done phi = {}/{}".format(p,q))
    
    # convert into ndarray
    phi = asarray(phi)
    eps = concatenate(eps) # necessary evil
    eps = asarray(eps)

    return phi, eps

def sum_over_kz(qmax=10,ucsize=1,kz_res=10,t=-1,M=2.3,D1=0.8,D2=0.5):
    """
    Sum over plane wave direction (z) to get correct butterfly pattern
    """
    # kz
    kz = linspace(-pi,pi,kz_res,endpoint=True)

    # initialize
    phi = []
    eps = []

    # fill up
    for k in kz:
        phi_k, eps_k = get_phis_eps(qmax=qmax,ucsize=ucsize,zu=k,t=t,M=M,D1=D1,D2=D2)
        phi.extend(phi_k)
        eps.extend(eps_k)
        print("DONE K = {:.3}".format(k))

    # convert into ndarray
    phi = asarray(phi)
    eps = asarray(eps)

    return phi, eps

# main
def main(qmax,ucsize=1,kz_res=10,t=-1,M=2.3,D1=0.8,D2=0.5):
    """
    Main function to plot SOTI butterfly
    """
    # sum_over_kz to get phi and eps
    phi, eps = sum_over_kz(qmax=qmax,ucsize=ucsize,kz_res=kz_res,t=t,M=M,D1=D1,D2=D2)
    savetxt("phi_soti.csv",phi,delimiter=',')
    savetxt("eps_soti.csv",eps,delimiter=',')

    return 

# run it
main(qmax=30,ucsize=3,kz_res=5,t=-1,M=2.3,D1=0.8,D2=0.5)

### Leo Goutte