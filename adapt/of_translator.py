import scipy
import numpy as np
from opt_einsum import contract
from openfermion.linalg import get_sparse_operator
import openfermion as of

    
def of_from_arrays(_0body, _1body, I, N_e, unpaired = 0, n_qubits = None, occupied = None, unoccupied = None):
    """Compute sparse matrices for ADAPT

    Parameters
    ----------
    _0body : float
        Nuclear repulsion energy
    _1body, I : numpy array
        1- and 2-electron integrals
    N_e : int
        Number of electrons 
    unpaired : int
        Number of unpaired electrons

    Returns
    -------
    hamiltonian, ref : scipy sparse matrix
        Hamiltonian, reference wfn
    N_qubits : int
        Number of qubits
    S2, Sz, number_operator : scipy sparse matrix
        S^2, S_z, and number operators 
    """
    _2body = I
    if n_qubits is None:
        n_qubits = _1body.shape[0]
    _1body = _1body[:n_qubits,:n_qubits]
    I = I[:n_qubits,:n_qubits,:n_qubits,:n_qubits]


    hamiltonian = of.ops.InteractionOperator(_0body, _1body, _2body)
    f_hamiltonian = of.ops.FermionOperator()
    for i in hamiltonian:
        f_hamiltonian += of.ops.FermionOperator(term = i, coefficient = hamiltonian[i])
    if occupied is not None:
         of.transforms.freeze_orbitals(f_hamiltonian, occupied, unoccupied = unoccupied, prune = True)

    hamiltonian = scipy.sparse.csr.csr_matrix(get_sparse_operator(f_hamiltonian, n_qubits = n_qubits).real)

    H = np.array(hamiltonian)

    #Build Number Operator for Internal Checks
    number_operator = of.ops.FermionOperator()
    for p in range(0, _1body.shape[0]):
        number_operator += of.ops.FermionOperator(term = ((p,1),(p,0)))

    number_operator = get_sparse_operator(number_operator, n_qubits = n_qubits).real


    #Build S^2 Operator for Internal Checks
    S2 = get_sparse_operator(of.hamiltonians.s_squared_operator(int(n_qubits/2)), n_qubits = n_qubits).real               
    #Build S_z Operator for Internal Checks
    Sz = get_sparse_operator(of.hamiltonians.sz_operator(int(n_qubits/2)), n_qubits = n_qubits).real
    #print("Assuming lexically first orbitals in chosen basis to be filled.")
    if unpaired == 0:
        ref = scipy.sparse.csc_matrix(of.jw_configuration_state(list(range(0, N_e)), _1body.shape[0])).T
    else:
        occs = list(range(0, N_e-unpaired))
        for i in range(0, unpaired):
            occs += [2*i + len(occs)]
        ref = scipy.sparse.csc_matrix(of.jw_configuration_state(occs, _1body.shape[0])).T
        

    return hamiltonian, ref, _1body.shape[0], S2, Sz, number_operator
