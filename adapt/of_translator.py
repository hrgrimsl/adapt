
import scipy
import numpy as np
from opt_einsum import contract
from openfermion.linalg import get_sparse_operator
import openfermion as of
def of_from_arrays(_0body, _1body, I, N_e, S_squared = None, S_z = None, unpaired = 0):
    _2body = I
    n_qubits = _1body.shape[0]
    hamiltonian = of.ops.InteractionOperator(_0body, _1body, _2body)
    hamiltonian = scipy.sparse.csr.csr_matrix(get_sparse_operator(hamiltonian, n_qubits = n_qubits).real)
    H = np.array(hamiltonian)
    #Build Number Operator for Internal Checks
    number_operator = of.ops.FermionOperator()
    for p in range(0, _1body.shape[0]):
        number_operator += of.ops.FermionOperator(term = ((p,1),(p,0)))

    number_operator = get_sparse_operator(number_operator, n_qubits = n_qubits).real


    #Build S^2 Operator for Internal Checks
    S2 = get_sparse_operator(of.hamiltonians.s_squared_operator(int(n_qubits/2)), n_qubits = n_qubits).real
    if S_squared != None:
        print("WARNING: Adding S^2 penalty to Hamiltonian")
        hamiltonian += (S2.dot(S2) - 2*S_squared*S2 + S_squared**2*scipy.sparse.identity(S2.shape[0]))
               
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
        
    #print("Reference Product State Energy:")
    #E_ref = ref.T.dot(hamiltonian).dot(ref)[0,0]
    #print(E_ref)
    #ci_spectrum, ci_vecs = np.linalg.eigh(hamiltonian.toarray())
    #print("CASCI Energy:")
    #print(ci_spectrum[0])
    #soln = ci_vecs[:,0]
    #print("Particle number")
    #print(round(soln.T.dot(number_operator.toarray()).dot(soln)))
    #print("S^2")
    #print('{:.3f}'.format(soln.T.dot(S2.toarray()).dot(soln)))
    #print("S_z")
    #print('{:.2f}'.format(soln.T.dot(Sz.toarray()).dot(soln)))
    return hamiltonian, ref, _1body.shape[0], S2, Sz, number_operator
