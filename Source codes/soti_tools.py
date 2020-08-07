# tools for SOTI butterfly plots and analysis

from numpy import *
from scipy.linalg import block_diag
import matplotlib.pyplot as plt
import scipy.sparse.linalg as spars


# sigma matrices
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

# harper matrix
# in what follows, the size of the system is *always* q
# alternatively, it could be a multiple of q by imposing gcd
def Harper_SOTI(p, q, mu = 0, nu = 0, zu = 0, t = -1, M = 2.3, D1 = 0.8, D2 = 0.5):
    # comment later
    # all energies are in units of |t|
    
    # size
    iq = int(q)
    
    # make block diagonals (same-site-hoppers)
    diags = M*s0_tz + t*s0_tz*cos(mu) + D1*sx_tx*sin(mu) + D2*s0_ty*cos(mu)
    diags_q = block_diag(*([diags]*iq)) # kron could be replaced by 
                                            # block_diag for large q
        
    cos_ms = [cos(2*pi*(p/q)*m - zu) for m in range(iq)]
    sin_ms = [sin(2*pi*(p/q)*m - zu) for m in range(iq)]
    diags_ms = kron(diag(cos_ms),t*s0_tz) + D1*kron(diag(sin_ms),sz_tx)
        # these are already filled out to q
        
    ssh = diags_q + diags_ms
    
    # make off-diagonal terms (next-site-hoppers)
    hop = (t*s0_tz + 1j*D1*sy_tx - D2*s0_ty)/2
    hop_dag = hop.conj().T
    
    hop_q = kron(diag(ones(iq-1), 1),hop)
    hop_dag_q = kron(diag(ones(iq-1), -1),hop_dag)
    
    nsh = hop_q + hop_dag_q
    
    # make boundary terms
    nsh[0:4,4*(iq-1):4*iq] = exp(1j*nu)*hop_dag 
    nsh[4*(iq-1):4*iq,0:4] = exp(-1j*nu)*hop
    
    # add em up
    Ha = ssh + nsh
    
    return Ha

# imshow of matrix entries
def matrix_form(p=1,q=10):
    H = Harper_SOTI(p,q)
    plt.figure(figsize=(10,10))
    plt.title(r"SOTI Harper matrix for $q = {} $".format(q), fontsize = 20, **futura)
    plt.imshow(abs(H), cmap = 'inferno')
    plt.xticks([])
    plt.yticks([])
    plt.show()

# eigenvalues of harper matrices
def eigs_harper(H):
    # computes eigs of Harper matrix
    # returns list of eigs
    # q = int(shape(H)[0])
    # spars scales better for large q
    eigs = linalg.eigvalsh(H) #, k = q, return_eigenvectors=False)
    
    return list(eigs)

# Greatest Common Divisor
def gcd(a, b):
    if b == 0: 
        return a
    else:
        return gcd(b, a % b)
# if gcd == 1 then the fraction is irreducible

# main function
# makes phis and energies
def main_single(qmax = 100, mu = 0, nu = 0, zu = 0, t = -1, M = 2.3, D1 = 0.8, D2 = 0.5):
    
    phi = []
    eps = []
    for q in range(1, qmax):
        for p in range(0, q): # all p s.t p/q < 1
            #if gcd(p,q) == 1:
            # add all possible phi (q copies of p/q and 1-p/q)
            phi.extend([p/q]*4*q + [(q-p)/q]*4*q)
            # compute eigs
            Harper = Harper_SOTI(p, q, mu, nu, zu, t, M, D1, D2)
            eigs_pq = eigs_harper(Harper)
            # add each eig twice for same reason as above (hermicity) # <- this may not apply anymore
            eps.extend(eigs_pq*2)
                
    # return phis and energies
    phi = asarray(phi)
    eps = asarray(eps)
    return phi, eps

def main(qmax = 100, kz_res = 5, mu = 0, nu = 0, t = -1, M = 2.3, D1 = 0.8, D2 = 0.5):
    """
    Phis and Eps to plot Butterfly for Schindler's SOTI
    """

    zus = linspace(-pi,pi,num=kz_res,endpoint=True)

    phi = []
    eps = []

    for zu in zus:
        phi_zu, eps_zu = main_single(qmax=qmax,mu=mu,nu=nu,zu=zu,t=t,M=M,D1=D1,D2=D2)
        phi.extend(phi_zu)
        eps.extend(eps_zu)

    return phi, eps


# spectrum creator
def spectrum(p, q, dimless_param, resolution = 100, 
    unit_cell_range = True, unit_cell_scale = 1):

    if unit_cell_range == True:
        ks = linspace(-pi/unit_cell_scale, pi/unit_cell_scale, resolution)
    elif unit_cell_range == False:
        ks = linspace(0, 2*pi/unit_cell_scale, resolution)

    k_ret = []
    Es = []

    for k in ks:
        A = Harper_SOTI(p = 50, q = q, zu = k)
        E = eigs_harper(A)
        Es.extend(E)
        k_ret.extend([k]*4*q)
    
    return k_ret, Es

# thin out arrays to plot
def thin_arrays(phi, eps, thin_size):
    
    indices = random.randint(len(phi)-1, size = int(thin_size))

    phi_thin = asarray([phi[index] for index in indices])
    eps_thin = asarray([eps[index] for index in indices])
    
    return phi_thin, eps_thin

# sums over all k and plots the resulting phi vs eps pattern
def sum_over_k(dimless_param, thin_size = 1e6, resolution = 100, unit_cell_scale = 1, 
    Thin = False):

    ks = linspace(-pi/unit_cell_scale,pi/unit_cell_scale,resolution)
    
    phi_ks = []
    eps_ks = []
    
    if dimless_param == 1:
        for k in ks:
            phi, eps = main_single(mu = k)
            phi_ks.extend(phi)
            eps_ks.extend(eps)
            
    elif dimless_param == 2:
        for k in ks:
            phi, eps = main_single(nu = k)
            phi_ks.extend(phi)
            eps_ks.extend(eps)
            
    elif dimless_param == 3:
        for k in ks:
            phi, eps = main_single(zu = k)
            phi_ks.extend(phi)
            eps_ks.extend(eps)

    if Thin == True:
        phi_ks_thin, eps_ks_thin = thin_arrays(phi_ks, eps_ks, thin_size)
        return phi_ks_thin, eps_ks_thin

    elif Thin == False:
        return phi_ks, eps_ks

# gets Es for given ks
def get_ke_spectrum(dimless_param, p, q, mu = 0, nu = 0, zu = 0, 
    resolution = 100,unit_cell_scale=1):
    Es = []
    k_ret = []
    ks = linspace(-pi/unit_cell_scale,pi/unit_cell_scale,resolution)

    # get energies for ks
    for k in ks:
        q = int(q)
        
        if dimless_param == 1:
            H = Harper_SOTI(p = p, q = q, mu = k, nu = nu, zu = zu)
        elif dimless_param == 2:
            H = Harper_SOTI(p = p, q = q, mu = mu, nu = k, zu = zu)
        elif dimless_param == 3:
            H = Harper_SOTI(p = p, q = q, mu = mu, nu = nu, zu = k)

        eigs_k = eigs_harper(H)
        Es.extend(eigs_k)
        k_ret.extend([k]*4*q)

    return k_ret, Es

futura = {'fontname':'Futura'}

# makes k vs e plots for given phis
def spectrum_plots(dimless_param, mu = 0, nu = 0, zu = 0,
    ps = [0,1,10,25,37,50], q = 100, resolution = 100, unit_cell_scale = 1):
    # ps must be a len six 1d array

    fig, ax = plt.subplots(nrows = 2, ncols = int(len(ps)/2), figsize = (20,10), 
        sharey = True, sharex = True)
    fig.subplots_adjust(wspace=0.1,hspace=0.25)
    
    for i in range(len(ps)):
        p_k = ps[i]
        ks, Es = get_ke_spectrum(dimless_param = dimless_param, mu = mu, nu = nu, zu = zu, 
            p = p_k, q = q, resolution = resolution, unit_cell_scale = unit_cell_scale)

        # fix indexing
        if i < 3:
            j = 0
        elif i >= 3:
            j = 1
            i -= 3
        
        # set labels
        if i == 0:
            ax[j][i].set_ylabel(r"${}$ - momentum".format(dimless_param), fontsize = 15, **futura)
        if j == 1: 
            ax[j][i].set_xlabel(r"Energy $\epsilon$", fontsize = 15, **futura)
        
        # plot
        ax[j][i].scatter(Es,ks,c='k',marker='.',s=1)
        #ax[j][i].set_xlim([-4,4]) # <- commented out for generality
        ax[j][i].set_title(r"$\Phi = {:.2}$".format(p_k/q), **futura)












