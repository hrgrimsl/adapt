from driver import *
from pyscf_backend import *
from of_translator import *
import system_methods as sm
import numpy as np


if __name__ == '__main__':
    #Example with some explanations
    
    #geometry in Angstroms
    geom = 'H 0 0 0; H 0 0 1; H 0 0 2; H 0 0 3'
    
    #basis set
    basis = "sto-3g"
    
    #I only expect rhf to work as a reference
    reference = "rhf"
        
    #Get molecular integrals, density matrix, MO coefficients, and HF energy    
    E_nuc, H_core, g, D, C, hf_energy = get_integrals(geom, basis, reference, chkfile = "scr.chk", read = False)
    
    #Compute number of electrons from density matrix
    N_e = int(np.trace(D))
    
    #Compute Hamiltonian, number of qubits, S^2, S_z, and number operators
    H, ref, N_qubits, S2, Sz, Nop = of_from_arrays(E_nuc, H_core, g, N_e)
    
    #Build system_data object for pool construction
    s = sm.system_data(H, ref, N_e, N_qubits)
    
    #Build pool of sparse operators and corresponding pool of strings    
    pool, v_pool = s.uccsd_pool(approach = 'vanilla')
    
    #Build Xiphos object to run ADAPT from
    #You will need to make directory 'test' for parameters and operators to be stored in
    xiphos = Xiphos(H, ref, "test", pool, v_pool, sym_ops = {"H": H, "S_z": Sz, "S^2": S2, "N": Nop}, verbose = "DEBUG")
    
    params = np.array([])
    ansatz = []
    #Run ADAPT with 300 initial guesses, plus recycled and HF guesses.  n != 1 corresponds to ADAPT^N calculation from the paper.
    xiphos.breadapt(params, ansatz, ref, Etol = 1e-12, guesses = 300, n = 1)


