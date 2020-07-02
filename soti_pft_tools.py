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

def unit_block(size, p, q, zu, t=-1, M = 2.3, D1 = 0.8, D2 = 0.5):
    """
    Block to go into soti_block diagonals (fixed x, varying y)
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

def soti_block(size, p , q, zu, t = -1, M = 2.3, D1 = 0.8, D2 = 0.5):
    """
    Block to go into H_SOTI diagonals (fixed y, varying x)
    """
    # put unit_blocks into diag
    
    # make blocks array with dims (size,4size,4size)
    blocks = zeros((size,4*size,4*size),dtype=complex) 
    
    # fill up
    xs = linspace(0,size,num=size) # for completeness
    for i in range(size):
        x = xs[i] # doesn't actually do anything
        blocks[i,:,:] = unit_block(size=size,p=p,q=q,zu=zu,t=t,
                                   M=M,D1=D1,D2=D2)
        
    # put in diagonal
    M_diags = ss.block_diag(blocks)
    
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

def H_SOTI(size, p, q, t = -1, M = 2.3, D1 = 0.8, D2 = 0.5):
    """
    SOTI Hamiltonian Fourier transformed in z and real in x,y
    """
    # q == size
    # condition on q and size
    if gcd(size,q) == 1:
        raise Exception("size must be a multiple of q")
        return 1

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

def kz_spectrum(p,q,kz_res=10,ucsize=1,t=-1,M=2.3,D1=0.8,D2=0.5):
    """
    Energies as a function of k_z
    """
    # kz
    kzs = linspace(-pi,pi,num=kz_res,endpoint=True)

    # for each kz, get Es from soti_block
    kz_ret = []
    Es = []

    # set the number of eigs returned
    newsize = q*ucsize
    num_eigs = int((4*newsize**2)/8) # <- let's try this for now and tweak if need be
    
    for kz in kzs:
        H_kz = soti_block(newsize,p,q,zu=kz,t=t,M=M,D1=D1,D2=D2)
        E_kz = ssl.eigsh(H_kz,k=num_eigs,sigma=0,return_eigenvectors=False)
        Es.extend(E_kz)
        kz_ret.extend([kz]*num_eigs)

    return kz_ret, Es

def get_phis_eps(qmax=10,ucsize=1,zu=0,t=-1,M=2.3,D1=0.8,D2=0.5):
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
            H_pq = soti_block(size=newsize,p=p,q=q,zu=zu,t=t,M=M,D1=D1,D2=D2)
            eigs_pq = ssl.eigsh(H_pq,k=H_dim,return_eigenvectors=False)
            eps.extend([eigs_pq]*2)
    
    # convert into ndarray
    phi = asarray(phi)
    eps = concatenate(eps)
    eps = asarray(eps)

    return phi, eps

def sum_over_kz(qmax=10,ucsize=1,kz_res=10,t=-1,M=2.3,D1=0.8,D2=0.5):
    """
    Sum over plane wave and gauge field direction (z) to get correct butterfly pattern
    """
    # kz
    kz = linspace(-pi,pi,num=kz_res,endpoint=True)

    # initialize
    phi = []
    eps = []

    # fill up
    for k in kz:
        phi_k, eps_k = get_phis_eps(qmax=qmax,ucsize=ucsize,zu=k,t=t,M=M,D1=D1,D2=D2)
        phi.extend(phi_k)
        eps.extend(eps_k)

    # convert into ndarray
    phi = asarray(phi)
    eps = asarray(eps)

    return phi, eps 

def spectrum_plots_kz(ps=[0,1,10],q=20,kz_res=10,ucsize=1,t=-1,M=2.3,D1=0.8,D2=0.5):
    """
    Plots of Energy as a function of k for the SOTI for various magnetic flux 
    """

    futura = {'fontname':'Futura'}

    # set up subplots
    fig, ax = plt.subplots(nrows = 1, ncols = int(len(ps)), figsize = (20,6), 
        sharey = True, sharex = True)
    fig.subplots_adjust(wspace=0.1,hspace=0.25) 

    # fill them up
    for i in range(len(ps)):
        pk = ps[i]
        ks, Eks = kz_spectrum(p=pk,q=q,kz_res=kz_res,ucsize=ucsize,t=t,M=M,D1=D1,D2=D2)
        
        # set labels
        if i == 0:
            ax[i].set_ylabel(r"$E/|t|$",fontsize = 15, **futura)
        ax[i].set_xlabel(r"$k$", fontsize = 15, **futura)

        # plot
        ax[i].scatter(ks,Eks,c='k',marker='.',s=5)
        ax[i].set_title(r"$\Phi = {:.2}$".format(pk/q), **futura)
        ax[i].set_ylim(-1,1) # range of interest

    return 

# too computationally expensive
def inspect_butterfly(size,q,phi_max,t=-1,M=2.3,D1=0.8,D2=0.5):
    """
    Zooms in on low Phi and near-fermi level energies for Schindler's SOTI butterfly.
    Contrary to other butterfly plot programs, the Phis here are evenly spaced. 
    """
    # make phis
    p_max = int(phi_max*q)
    phi = []
    eps = []
    print("hello")
    # number of eigenvalues near e = 0
    num_eigs = int(4*size)

    # set up zus
    #myzus = linspace(-pi,pi,num=5,endpoint=True)

    # get energies for phis
    #for zu in zus:
    for p in range(p_max):
        H_pq = soti_block(size=size,p=p,q=q,zu=0,t=t,M=M,D1=D1,D2=D2)
        print("got H")
        eps_pq = ssl.eigsh(H_pq,k=num_eigs,sigma=0,return_eigenvectors=False)
        print("got eigs")
        phi.extend([p/q]*num_eigs) # no need to get (q-p)/q
        eps.extend(eps_pq)

    return phi, eps