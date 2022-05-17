"""
Unit and regression test for the adapt package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest
import os
import adapt
from adapt.driver import *
from adapt.pyscf_backend import *
from adapt.of_translator import *

def test_adapt_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "adapt" in sys.modules

def test_adapt_vqe():
    """Test ADAPT on H2."""
    if os.path.exists('test') == False:
        os.makedirs('test')
    geom = 'H 0 0 0; H 0 0 1'
    basis = "sto-3g"
    reference = "rhf"
    E_nuc, H_core, g, D, C, hf_energy = get_integrals(geom, basis, reference, chkfile = "scr.chk", read = False)


    N_e = int(np.trace(D))


    H, ref, N_qubits, S2, Sz, Nop = of_from_arrays(E_nuc, H_core, g, N_e)


    s = sm.system_data(H, ref, N_e, N_qubits)

    pool, v_pool = s.uccsd_pool(approach = 'vanilla')

    xiphos = Xiphos(H, ref, "test", pool, v_pool, sym_ops = {"H": H, "S_z": Sz, "S^2": S2, "N": Nop}, verbose = "DEBUG")
    params = np.array([])
    ansatz = []


    error = xiphos.breadapt(params, ansatz, ref, Etol = 1e-8, guesses = 0, hf = False, n = 1)
    assert error < 1e-8
  
