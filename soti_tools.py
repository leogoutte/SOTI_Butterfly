# tools for SOTI butterfly plots and analysis

import numpy as np
from scipy.linalg import block_diag
import matplotlib.pyplot as plt

# sigma matrices
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

# harper matrice
def Harper_SOTI(p, q, mu = 0, nu = 0, zu = 0, M = 2.3, D1 = 0.8, D2 = 0.5):
    # comment later
    # all energies are in units of |t|
    
    # size
    iq = int(q)
    
    # define kroneckers of pauli matrices
    s0_ty = np.kron(sig(0),sig(2))
    s0_tz = np.kron(sig(0),sig(3))
    sx_tx = np.kron(sig(1),sig(1))
    sy_tx = np.kron(sig(2),sig(1))
    sz_tx = np.kron(sig(3),sig(1))
    
    # make block diagonals (same-site-hoppers)
    diags = M*s0_tz + s0_tz*np.cos(mu) + D1*sx_tx*np.sin(mu) + D2*s0_ty*np.cos(mu)
    diags_q = block_diag(*([diags]*iq)) # np.kron could be replaced by 
                                            # np.block_diag for large q
        
    cos_ms = [np.cos(2*np.pi*(p/q)*m - zu) for m in range(iq)] # <- could be mistake here
    sin_ms = [np.sin(2*np.pi*(p/q)*m - zu) for m in range(iq)]
    diags_ms = np.kron(np.diag(cos_ms),s0_tz) + D1*np.kron(np.diag(sin_ms),sz_tx)
        # these are already filled out to q
        
    ssh = diags_q + diags_ms
    
    # make off-diagonal terms (next-site-hoppers)
    hop = (s0_tz + 1j*D1*sy_tx - D2*s0_ty)/2
    hop_dag = hop.conj().T
    
    hop_q = np.kron(np.diag(np.ones(iq-1), 1),hop)
    hop_dag_q = np.kron(np.diag(np.ones(iq-1), -1),hop_dag)
    
    nsh = hop_q + hop_dag_q
    
    # make boundary terms
    nsh[0:4,4*(iq-1):4*iq] = np.exp(1j*nu)*hop_dag # <- could be mistake here
    nsh[4*(iq-1):4*iq,0:4] = np.exp(-1j*nu)*hop
    
    # add em up
    Ha = ssh + nsh
    
    return Ha

# eigenvalues of harper matrices
def eigs_harper(H):
    # computes eigs of Harper matrix
    # returns list of eigs
    
    eigs = np.linalg.eigvalsh(H)
    
    return list(eigs

# Greatest Common Divisor
def gcd(a, b):
    if b == 0: 
    	return a
    return gcd(b, a % b)
# if gcd == 1 then the fraction is irreducible

# main function
# makes phis and energies
def main(qmax = 100, mu = 0, nu = 0, zu = 0, 
	M = 2.3, D1 = 0.8, D2 = 0.5):

    phi = []
    eps = []
    for q in range(1, qmax):
        for p in range(0, q): # all p s.t p/q < 1
            if gcd(p,q) == 1:
                # add all possible phi (q copies of p/q and 1-p/q)
                phi.extend([p/q]*4*q + [(q-p)/q]*4*q)
                # compute eigs
                Harper = Harper_SOTI(p, q, mu, nu, zu, M, D1, D2)
                eigs_pq = eigs_harper(Harper)
                # add each eig twice for same reason as above (hermicity) # <- this may not apply anymore
                eps.extend(eigs_pq*2)
                
    # return phis and energies
    phi = np.asarray(phi)
    eps = np.asarray(eps)
    return phi, 

# spectrum creator
def spectrum(p, q, dimless_param, resolution = 100, 
	unit_cell_range = True, unit_cell_scale = 1):

	if unit_cell_range == True:
		ks = np.linspace(-np.pi/unit_cell_scale, np.pi/unit_cell_scale, resolution)
	elif unit_cell_range == False:
		ks = np.linspace(0, 2*np.pi/unit_cell_scale, resolution)

	k_ret = []
	Es = []

	for k in ks:
    	A = Harper_SOTI(p = 50, q = qk, zu = k)
    	E = eigs_harper(A)
    	Es.extend(E)
    	k_ret.extend([k]*4*qk)
    
	return k_ret, Es
















def spectrum_plots():
    # makes k vs energy plots
    


















