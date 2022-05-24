import sys
import pytest
import os
import adapt
from adapt.driver import *
from adapt.pyscf_backend import *
from adapt.of_translator import *

def run_example():
    """Test ADAPT on H4."""
    if os.path.exists('test') == False:
        os.makedirs('test')
    geom = 'H 0 0 0; H 0 0 1; H 0 0 2; H 0 0 3'
    basis = "sto-3g"
    reference = "rhf"
    
    #Compute molecular integrals, etc.
    E_nuc, H_core, g, D, C, hf_energy = get_integrals(geom, basis, reference, chkfile = "scr.chk", read = False)
    
    #Compute number of electrons
    N_e = int(np.trace(D))
    
    #Compute sparse matrix reps
    H, ref, N_qubits, S2, Sz, Nop = of_from_arrays(E_nuc, H_core, g, N_e)
    
    #Build system data and get pool from it
    s = sm.system_data(H, ref, N_e, N_qubits)
    pool, v_pool = s.uccsd_pool(approach = 'vanilla')
    
    #Build 'xiphos' object (essentially ADAPT class)
    xiphos = Xiphos(H, ref, "test", pool, v_pool, sym_ops = {"H": H, "S_z": Sz, "S^2": S2, "N": Nop})
    params = np.array([])
    ansatz = []
    
    #Run ADAPT^1
    error = xiphos.breadapt(params, ansatz, ref, Etol = 1e-8, guesses = 0, hf = False, n = 1, threads = 1)

if __name__ == "__main__":
   run_example()


