import adapt.system_methods as sm
from pyscf import fci, gto, scf, mcscf
from pyscf.lib import logger
import numpy as np
from opt_einsum import contract

def get_integrals(geometry, basis, reference, charge = 0, spin = 0, read = False, chkfile = 'chk', feed_C = False, scf_grad = 1e-14, active = None):    
    """Function to get the pyscf integrals and stuff to feed to OpenFermion

    Parameters
    ----------
    geometry, basis, reference : string
         geometry, basis set, and reference.  (I don't trust any reference besides rhf for now...)
    charge, spin : int
         charge and spin of the system to feed pyscf
    read : bool
         Try to read from chkfile as an initial HF guess?
    chkfile : string
         chkfile for pyscf
    feed_C : bool/numpy array
         Either False or a set of MO coefficients to use instead of the HF ones
    scf_grad : float
         scf gradient tightness in pyscf
    active: tuple
         number of active spatial orbitals and number of active electrons respectively

    Returns
    -------
    E_nuc : float
        Nuclear repulsion energy
    H_core, g : numpy array
        1- and 2- electron integrals
    D : numpy array
        Density matrix
    C : numpy array
        MO coefficients
    hf_energy : float
        HF energy
    """
    mol = gto.M(atom = geometry, basis = basis, spin = spin, charge = charge, verbose = True)
    mol.verbose = 4
    mol.symmetry = False
    mol.max_memory = 8e3
    mol.build()
    if reference == 'rhf':
        mf = scf.RHF(mol)
    elif reference == 'rohf':
        mf = scf.ROHF(mol)
    elif reference == 'uhf':
        mf = scf.UHF(mol)
    else:
        print('Reference not understood.')
    mf.chkfile = chkfile
    mf.conv_tol_grad = scf_grad
    mf.max_cycle = 10000
    mf.verbose = 4
    mf.conv_check = True
    if read == True:
        mf.init_guess = 'chkfile'
    else:
        mf.init_guess = 'atom'

    hf_energy = mf.kernel()
    assert mf.converged == True
    mo_occ = mf.mo_occ
    C = mf.mo_coeff
    mo_occ = mf.mo_occ
    Oa = 0
    Ob = 0
    Va = 0
    Vb = 0
    if reference != "uhf":
        Ca = Cb = mf.mo_coeff
        mo_a = np.zeros(len(mo_occ))
        mo_b = np.zeros(len(mo_occ))
        for i in range(0, len(mo_occ)):
            if mo_occ[i] > 0:
                mo_a[i] = 1
                Oa += 1
            else:
                Va += 1
            if mo_occ[i] > 1:
                mo_b[i] = 1
                Ob += 1
            else:
                Vb += 1

    else:
        Ca = mf.mo_coeff[0]
        Cb = mf.mo_coeff[1]
        mo_a = np.zeros(len(mo_occ[0]))
        mo_b = np.zeros(len(mo_occ[1]))
        for i in range(0, len(mo_occ[0])):
            if mo_occ[0][i] == 1:
                mo_a[i] = 1
                Oa += 1
            else:
                Va += 1
        for i in range(0, len(mo_occ[1])):
            if mo_occ[1][i] == 1:
                mo_b[i] = 1
                Ob += 1
            else:
                Vb += 1
    

    if feed_C != False:
        print("Loading C")
        Ca = Cb = np.load(feed_C)

    Da = np.diag(mo_a)
    Db = np.diag(mo_b)
    S = mol.intor('int1e_ovlp_sph')
    E_nuc = mol.energy_nuc()
    H_core = mol.intor('int1e_nuc_sph') + mol.intor('int1e_kin_sph')
    I = mol.intor('int2e_sph')
    Ha = Ca.T.dot(H_core).dot(Ca)
    Hb = Cb.T.dot(H_core).dot(Cb)
    Iaa = contract('pqrs,pi,qj,rk,sl->ikjl', I, Ca, Ca, Ca, Ca)
    Iab = contract('pqrs,pi,qj,rk,sl->ikjl', I, Ca, Ca, Cb, Cb)
    Iba = contract('pqrs->qpsr', Iab)
    Ibb = contract('pqrs,pi,qj,rk,sl->ikjl', I, Cb, Cb, Cb, Cb)

    Ja = contract('pqrs,qs->pr', Iaa, Da)+contract('pqrs,qs->pr', Iab, Db)
    Jb = contract('pqrs,qs->pr', Ibb, Db)+contract('pqrs,pr->qs', Iab, Da)
    Ka = contract('pqrs,qr->ps', Iaa, Da)
    Kb = contract('pqrs,ps->qr', Ibb, Db)

    Fa = Ha + Ja - Ka
    Fb = Hb + Jb - Kb
    manual_energy = E_nuc + .5*contract('pq,pq', Ha + Fa, Da) + .5*contract('pq,pq', Hb + Fb, Db)

    A = np.array([[1,0],[0,0]])
    B = np.array([[0,0],[0,1]])
    AA = contract('ia,jb->ijab', A, A)
    BB = contract('ia,jb->ijab', B, B)
    AB = contract('ia,jb->ijab', A, B)
    BA = contract('ia,jb->jiba', A, B)
    
    H_core = np.kron(Ha, A) + np.kron(Hb, B)
    D = np.kron(Da, A) + np.kron(Db, B)
    C = np.kron(Ca, A) + np.kron(Cb, B)
    g = np.kron(Iaa, AA) + np.kron(Iab, AB) + np.kron(Iba, BA) + np.kron(Ibb, BB)

    g -= contract('pqrs->pqsr', g)
    g *= -.25
    if active is None:
        cisolver = fci.FCI(mol, mf.mo_coeff)
        print("PYSCF FCI:")
        print(cisolver.kernel(verbose=logger.DEBUG)[0])
    else:
        mycas = mcscf.CASCI(mf, active[0], active[1])
        casci = mycas.kernel(verbose=logger.DEBUG)
        print("PYSCF CASCI:")
        print(casci[0])
    return E_nuc, H_core, g, D, C, hf_energy

def get_F(geometry, basis, reference, charge = 0, spin = 0, feed_C = False):
    mol = gto.M(atom = geometry, basis = basis, spin = spin, charge = charge)
    mol.symmetry = False
    mol.max_memory = 8e3
    mol.build()
    if reference == 'rhf':
        mf = scf.RHF(mol)
    elif reference == 'rohf':
        mf = scf.ROHF(mol)
    elif reference == 'uhf':
        mf = scf.UHF(mol)
    else:
        print('Reference not understood.')
    mf.conv_tol = 1e-12
    mf.max_cycle = 1000
    mf.verbose = 0
    mf.conv_check = True
    hf_energy = mf.kernel()
    assert mf.converged == True
    mo_occ = mf.mo_occ
    C = mf.mo_coeff
    mo_occ = mf.mo_occ
    Oa = 0
    Ob = 0
    Va = 0
    Vb = 0
    if reference != "uhf":
        Ca = Cb = mf.mo_coeff
        mo_a = np.zeros(len(mo_occ))
        mo_b = np.zeros(len(mo_occ))
        for i in range(0, len(mo_occ)):
            if mo_occ[i] > 0:
                mo_a[i] = 1
                Oa += 1
            else:
                Va += 1
            if mo_occ[i] > 1:
                mo_b[i] = 1
                Ob += 1
            else:
                Vb += 1

    else:
        Ca = mf.mo_coeff[0]
        Cb = mf.mo_coeff[1]
        mo_a = np.zeros(len(mo_occ[0]))
        mo_b = np.zeros(len(mo_occ[1]))
        for i in range(0, len(mo_occ[0])):
            if mo_occ[0][i] == 1:
                mo_a[i] = 1
                Oa += 1
            else:
                Va += 1
        for i in range(0, len(mo_occ[1])):
            if mo_occ[1][i] == 1:
                mo_b[i] = 1
                Ob += 1
            else:
                Vb += 1
    if feed_C != False:
        print("Loading C")
        Ca = Cb = np.load(feed_C)
    Da = np.diag(mo_a)
    Db = np.diag(mo_b)
    S = mol.intor('int1e_ovlp_sph')
    E_nuc = mol.energy_nuc()
    H_core = mol.intor('int1e_nuc_sph') + mol.intor('int1e_kin_sph')
    I = mol.intor('int2e_sph')
    Ha = Ca.T.dot(H_core).dot(Ca)
    Hb = Cb.T.dot(H_core).dot(Cb)
    Iaa = contract('pqrs,pi,qj,rk,sl->ikjl', I, Ca, Ca, Ca, Ca)
    Iab = contract('pqrs,pi,qj,rk,sl->ikjl', I, Ca, Ca, Cb, Cb)
    Iba = contract('pqrs->qpsr', Iab)
    Ibb = contract('pqrs,pi,qj,rk,sl->ikjl', I, Cb, Cb, Cb, Cb)

    Ja = contract('pqrs,qs->pr', Iaa, Da)+contract('pqrs,qs->pr', Iab, Db)
    Jb = contract('pqrs,qs->pr', Ibb, Db)+contract('pqrs,pr->qs', Iab, Da)
    Ka = contract('pqrs,qr->ps', Iaa, Da)
    Kb = contract('pqrs,ps->qr', Ibb, Db)

    Fa = Ha + Ja - Ka
    Fb = Hb + Jb - Kb
    manual_energy = E_nuc + .5*contract('pq,pq', Ha + Fa, Da) + .5*contract('pq,pq', Hb + Fb, Db)


    A = np.array([[1,0],[0,0]])
    B = np.array([[0,0],[0,1]])
    AA = contract('ia,jb->ijab', A, A)
    BB = contract('ia,jb->ijab', B, B)
    AB = contract('ia,jb->ijab', A, B)
    BA = contract('ia,jb->jiba', A, B)
    
    H_core = np.kron(Ha, A) + np.kron(Hb, B)
    D = np.kron(Da, A) + np.kron(Db, B)
    C = np.kron(Ca, A) + np.kron(Cb, B)
    g = np.kron(Iaa, AA) + np.kron(Iab, AB) + np.kron(Iba, BA) + np.kron(Ibb, BB)
    F = np.kron(Fa, A) + np.kron(Fb, B)
    g -= contract('pqrs->pqsr', g)
    g *= -.25
    return F

def freeze_core(E_nuc, H, I, D, N_c):
    D_core = D[:N_c,:N_c]
    J_core = contract('pqrs,rs->pq', I[:N_c, :N_c, :N_c, :N_c], D_core) 
    K_core = contract('psrq,rs->pq', I[:N_c, :N_c, :N_c, :N_c], D_core)
    zero_body = E_nuc + .5*contract('pq,pq->', (2*H[:N_c,:N_c] + J_core - K_core), D_core)
    J_mix = contract('pqrs,rs->pq', I[N_c:, N_c:, :N_c, :N_c], D_core)
    K_mix = contract('psrq,rs->pq', I[N_c:, :N_c, :N_c, N_c:], D_core)
    one_body = H[N_c:, N_c:] + J_mix - K_mix   
    two_body = I[N_c:, N_c:, N_c:, N_c:] 
    return zero_body, one_body, two_body, D[N_c:, N_c:]

def rotate(H, I, R, rotate_rdm = False):
    #Typically you'll do 2 rotations- one to rotate into the AO basis, and another to rotate into the new basis
    H = contract('pq,pi,qj->ij', H, R, R)
    I = contract('pqrs,pi,qj,rk,sl->ijkl', I, R, R, R, R)
    return H, I
