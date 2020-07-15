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

def get_wavefunctions(p, q, num_eigs, ucsize=1, zu=0, t = -1, M = 2.3, D1 = 0.8, D2 = 0.5):
    """
    Get the wavefunctions of the 4 conducting states for the SOTI.
    """

    # set params
    energy_centre = 0

    # Harper matrix 
    H_pq = soti_block(size=q*ucsize,p=p,q=q,zu=zu,t=t,M=M,D1=D1,D2=D2)
    eps_pq, wvfcts_pq = ssl.eigsh(H_pq,k=num_eigs,sigma=energy_centre)

    return eps_pq, wvfcts_pq

def normalize_prob(stuff):
    """
    Simple normalization function.
    """
    sums = np.sum(stuff,axis=0)
    return np.divide(stuff,sums)

def wavefunction_to_probability(waves, spin_nos=4):
    """
    Transforms a wavefunction (probability amplitude) into a probability
    distribution.
    """

    # get probability distribution
    waves_abs = np.abs(waves)**2
    vec_nos = waves_abs.shape[1]

    # dimensions of positions
    pos_dim = int((waves.shape[0]/spin_nos)**(1/2))

    # batch over spin components
    batched_size = int(waves.shape[0]/spin_nos)
    prob_pos = np.zeros((batched_size,vec_nos), dtype = float)
    for i in range(batched_size):
        batched_sum = np.sum(waves_abs[i:i+spin_nos,:],axis=0)
        prob_pos[i,:] = batched_sum

    

    # extract y dependence
    prob_y = prob_pos[0:pos_dim,:] # possible improvement: avg over
                                   # all normed batches of prob_y 
    prob_y_normed = normalize_prob(prob_y)

    # extract x dependence
    prob_x = np.zeros((pos_dim,vec_nos),dtype=float)
    for i in range(pos_dim):
        x_vals = prob_pos[i:i+pos_dim,:]/prob_y_normed # array of all the same x values
        prob_x[i,:] = np.mean(x_vals,axis=0)
    prob_x_normed = normalize_prob(prob_x)

    # return the (pos_dim,vec_nos) arrays for x and y probabilities
    return prob_x_normed, prob_y_normed