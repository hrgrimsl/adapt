#Gives molecule class w/ methods for deriving pools, etc.
import openfermion as of
import scipy
import re
#import psi4
import numpy as np
import copy

np.set_printoptions(edgeitems=10)
np.core.arrayprint._line_width = 180
from opt_einsum import contract
class system_data:
    def __init__(self, H, ref, N_e, N_qubits):
        """Initialize system data object
        Parameters
        ----------
        H, ref : scipy sparse matrix
             Hamiltonian and reference wfn.
        N_e, N_qubits: int
             Number of electrons and number of qubits
        
        Returns
        -------
        None
        """
        self.N_qubits = N_qubits
        self.ref = ref
        self.H = H
        self.N_e = N_e
        self.pool = []
        #energies, wfns = np.linalg.eigh(H.toarray())
        #self.ci_energy = energies[0]
        #self.ci_soln = wfns[:,0]
        #self.hf_energy = self.ref.T.dot(self.H).dot(self.ref)[0,0]                

    def recursive_qubit_op(self, op, qubit_index):
        if qubit_index == self.N_qubits-1:
            return [op, op + ' X' + str(qubit_index), op + ' Y' + str(qubit_index), op + ' Z' + str(qubit_index)]
        else:
            return self.recursive_qubit_op(op, qubit_index+1) + self.recursive_qubit_op(op + ' X' + str(qubit_index), qubit_index+1) + self.recursive_qubit_op(op + ' Y' + str(qubit_index), qubit_index+1) + self.recursive_qubit_op(op + ' Z' + str(qubit_index), qubit_index+1)
        
    def choose_next(self, set_of_lists, cur_list, k):
        if len(cur_list) == k:
            set_of_lists += [cur_list]
        else:
            if len(cur_list) == 0:
                floor = 0
            else:
                floor = max(cur_list)+1
            for i in range(floor, self.N_qubits):
                self.choose_next(set_of_lists, cur_list+[i], k)

    def choose_paulis(self, paulis, sub_list, k):
        if len(sub_list) == k:
            paulis += [sub_list]
        else:
            for let in ['X', 'Y', 'Z']:
                self.choose_paulis(paulis, sub_list + [let], k)
                        
    def tang_pool(self):
        """Pool from the qubit-ADAPT paper

        Returns
        -------
        jw_pool, fermi_ops : list
             List of sparse matrix operators and their verbal representations respectively
        """

        #this gives weird warnings...
        M = int(self.N_qubits/2)
        N = int(self.N_e/2)
        #build Spin-adapted GSD pool of fermionic ops
        sq_pool = []
        
        for p in range(0, M):
            pa = 2*p
            pb = 2*p+1
            for q in range(p+1, M):
                qa = 2*q
                qb = 2*q+1
                sq_pool.append(1/np.sqrt(2)*of.ops.FermionOperator(((pa, 1), (qa, 0)))+1/np.sqrt(2)*of.ops.FermionOperator(((pb, 1), (qb, 0))))
                sq_pool[-1] -= of.utils.hermitian_conjugated(sq_pool[-1])
        
        pq = -1
        for p in range(0, M):
            pa = 2*p
            pb = 2*p+1
            for q in range(p, M):
                qa = 2*q
                qb = 2*q+1

                pq += 1

                rs = -1
                for r in range(0, M):
                    ra = 2*r
                    rb = 2*r+1
                    for s in range(r, M):
                        sa = 2*s
                        sb = 2*s+1
                        
                        rs += 1
                        
                        if pq > rs: 
                            continue
                        
                        termA = of.ops.FermionOperator(((ra, 1), (pa, 0), (sa, 1), (qa, 0)), 2 / np.sqrt(12))
                        termA += of.ops.FermionOperator(((rb, 1), (pb, 0), (sb, 1), (qb, 0)), 2 / np.sqrt(12))
                        termA += of.ops.FermionOperator(((ra, 1), (pa, 0), (sb, 1), (qb, 0)), 1 / np.sqrt(12))
                        termA += of.ops.FermionOperator(((rb, 1), (pb, 0), (sa, 1), (qa, 0)), 1 / np.sqrt(12))
                        termA += of.ops.FermionOperator(((ra, 1), (pb, 0), (sb, 1), (qa, 0)), 1 / np.sqrt(12))
                        termA += of.ops.FermionOperator(((rb, 1), (pa, 0), (sa, 1), (qb, 0)), 1 / np.sqrt(12))

                        termB = of.ops.FermionOperator(((ra, 1), (pa, 0), (sb, 1), (qb, 0)), 1 / 2.0)
                        termB += of.ops.FermionOperator(((rb, 1), (pb, 0), (sa, 1), (qa, 0)), 1 / 2.0)
                        termB += of.ops.FermionOperator(((ra, 1), (pb, 0), (sb, 1), (qa, 0)), -1 / 2.0)
                        termB += of.ops.FermionOperator(((rb, 1), (pa, 0), (sa, 1), (qb, 0)), -1 / 2.0)
                        
                        termA -= of.utils.hermitian_conjugated(termA)
                        termB -= of.utils.hermitian_conjugated(termB)
                        
                        termA = of.transforms.normal_ordered(termA)
                        termB = of.transforms.normal_ordered(termB)
                        if termA.many_body_order() > 0:
                            sq_pool.append(termA)
                        if termB.many_body_order() > 0:
                            sq_pool.append(termB)
        #normalization shouldn't matter here
        print(f"{len(sq_pool)} operators in the UCCGSD pool.")
        print(f"Extracting Pauli strings...")
        n = 2*M
        pool_vec = np.zeros((4 **n,))

        for i in sq_pool:
            pauli = of.transforms.jordan_wigner(i)
            for line in pauli.terms:
                line = str(line)
                # print(line)
                Bin = np.zeros((2 * n,), dtype=int)
                X_pat_1 = re.compile("(\d{,2}), 'X'")
                X_1 = X_pat_1.findall(line)
                if X_1:
                    for i in X_1:
                        k = int(i)
                        Bin[n + k] = 1
                Y_pat_1 = re.compile("(\d{,2}), 'Y'")
                Y_1 = Y_pat_1.findall(line)
                if Y_1:
                    for i in Y_1:
                        k = int(i)
                        Bin[n + k] = 1
                        Bin[k] = 1
                Z_pat_1 = re.compile("(\d{,2}), 'Z'")
                Z_1 = Z_pat_1.findall(line)
                if Z_1:
                    for i in Z_1:
                        k = int(i)
                        Bin[k] = 1
                # print(Bin)
                index = int("".join(str(x) for x in Bin), 2)
                # print("index", index)
                pool_vec[index] = int(1)

        nz = np.nonzero(pool_vec)[0]

        print("Pauli Pool Size:", len(pool_vec[nz]))
        fermi_ops = []

        m = 2*n
        jw_pool = []
        for i in nz:
            p = int(i)
            bi = bin(p)
            b_string = [int(j) for j in bi[2:].zfill(m)]
            pauli_string = ''
            flip = []
            for k in range(n):
                if b_string[k] == 0:
                    if b_string[k + n] == 1:
                        pauli_string += 'X%d ' % k
                        flip.append(k)
                if b_string[k] == 1:
                    if b_string[k + n] == 1:
                        pauli_string += 'Y%d ' % k
                        flip.append(k)
            flip.sort()
            z_string = list(range(flip[0] + 1,flip[1]))
            if len(flip) == 4:
                for i in range(flip[2] + 1, flip[3]):
                    z_string.append(i)
            #print("Z string:", z_string)
            for i in z_string:
                b_string[i] += 1
                b_string[i] = b_string[i] % 2
            for k in range(n):
                if b_string[k] == 1:
                    if b_string[k + n] == 0:
                        pauli_string += 'Z%d ' % k
            A = of.ops.QubitOperator(pauli_string, 0 + 1j)
            fermi_ops.append(A)
            jw_pool.append(of.get_sparse_operator(A, self.N_qubits))
        return jw_pool, fermi_ops

    def full_qubit_pool(self):
        pool = []
        pool += self.recursive_qubit_op("", 0)
        assert(len(pool) == 4**self.N_qubits)
        self.pool += [i for i in pool if len(re.findall("Y", i))%2 == 1]
        return [of.get_sparse_operator(1j * of.ops.QubitOperator(i), self.N_qubits) for i in self.pool]

    def k_qubit_pool(self, k):
        indices = []
        self.choose_next(indices, [], k)
        paulis = []
        self.choose_paulis(paulis, [], k)
        pool = []
        for i in indices:
            for j in paulis:
                string = str(j[0])+str(i[0])
                for l in range(1, len(i)):
                    string += " "+str(j[l])+str(i[l])
                pool.append(string)
        pool = [i for i in pool if len(re.findall("Y", i))%2 == 1]
        self.pool += pool 
        return [(of.get_sparse_operator(1j * of.ops.QubitOperator(i), self.N_qubits)).real for i in pool], pool

    def kup_pool(self):
        N_qubits = self.N_qubits
        N_e = self.N_e
        pool = []
        v_pool = []
        
        M = int(N_qubits/2)
        N = int(N_e/2)
        for p in range(0, M):
            for q in range(p+1, M):
                pool.append(of.ops.FermionOperator(str(2*q)+'^ '+str(2*p), 1))
                v_pool.append(f"{2*p}->{2*q}")
                pool.append(of.ops.FermionOperator(str(2*q+1)+'^ '+str(2*p+1), 1))
                v_pool.append(f"{2*p+1}->{2*q+1}")
                pool.append(of.ops.FermionOperator(str(2*q+1)+'^ '+str(2*q)+'^ '+str(2*p)+' '+str(2*p+1), 1))
                v_pool.append(f"{2*p},{2*p+1}->{2*q},{2*q+1}")
        jw_pool = [scipy.sparse.csr.csr_matrix(of.linalg.get_sparse_operator(i, n_qubits = N_qubits).real) for i in pool]
        jw_pool2 = [A-A.T.conjugate() for A in jw_pool]
        print("Operators in k-Up pool:")        
        return jw_pool2, v_pool
        
    def afi_pool(self):
        N_qubits = self.N_qubits
        N_e = self.N_e
        pool = []
        v_pool = []

        M = int(N_qubits/2)
        N = int(N_e/2)
        for i in range(0, N):
            for a in range(N, M):
                pool.append(of.ops.FermionOperator(str(2*a)+'^ '+str(2*i), 1/np.sqrt(2))+of.ops.FermionOperator(str(2*a+1)+'^ '+str(2*i+1), 1/np.sqrt(2)))
                v_pool.append(f"{i}->{a}")
                for j in range(i, N):
                    for b in range(a, M):
                        if i == j and a == b:
                            pool.append(of.ops.FermionOperator(str(2*b+1)+'^ '+str(2*a)+'^ '+str(2*i)+' '+str(2*j+1), 1))
                            v_pool.append(f"{i}{j}->{a}{b}")
                        elif i == j:
                            pool.append(of.ops.FermionOperator(str(2*b+1)+'^ '+str(2*a)+'^ '+str(2*i)+' '+str(2*j+1), 1/np.sqrt(2)) + of.ops.FermionOperator(str(2*a+1)+'^ '+str(2*b)+'^ '+str(2*i)+' '+str(2*j+1), 1/np.sqrt(2)))
                            v_pool.append(f"{i}{j}->{a}{b}")
                        elif a == b:
                            pool.append(of.ops.FermionOperator(str(2*b+1)+'^ '+str(2*a)+'^ '+str(2*i)+' '+str(2*j+1), 1/np.sqrt(2)) + of.ops.FermionOperator(str(2*b+1)+'^ '+str(2*a)+'^ '+str(2*j)+' '+str(2*i+1), 1/np.sqrt(2)))
                            v_pool.append(f"{i}{j}->{a}{b}")
                        else:
                            pool.append(of.ops.FermionOperator(str(2*b)+'^ '+str(2*a)+'^ '+str(2*i)+' '+str(2*j), 1/2)+ of.ops.FermionOperator(str(2*b+1)+'^ '+str(2*a)+'^ '+str(2*i)+' '+str(2*j+1), 1/2)+ of.ops.FermionOperator(str(2*a+1)+'^ '+str(2*b)+'^ '+str(2*j)+' '+str(2*i+1), 1/2)+ of.ops.FermionOperator(str(2*b+1)+'^ '+str(2*a+1)+'^ '+str(2*i+1)+' '+str(2*j+1), 1/2))
                            v_pool.append(f"{i}{j}->{a}{b}")
            
        #Adding normalization
        for i in range(0, len(pool)):
             op = copy.copy(pool[i])
             op -= of.hermitian_conjugated(op)
             op = of.normal_ordered(op)
             assert(op.many_body_order() > 0)

        jw_pool = [scipy.sparse.csr.csr_matrix(of.linalg.get_sparse_operator(i, n_qubits = N_qubits).real) for i in pool]
        self.pool = pool
        print("Operators in pool:")
        print(len(pool)) 
        return jw_pool, v_pool
        

    def grimsley_pool(self):
        N_qubits = self.N_qubits
        N_e = self.N_e
        pool = []
        v_pool = []

        M = int(N_qubits/2)
        N = int(N_e/2)
        for i in range(0, N):
            for a in range(N, M):
                pool.append(of.ops.FermionOperator(str(2*a)+'^ '+str(2*i), 1/np.sqrt(2))+of.ops.FermionOperator(str(2*a+1)+'^ '+str(2*i+1), 1/np.sqrt(2)))
                v_pool.append(f"{i}->{a}")
                for j in range(i, N):
                    for b in range(a, M):
                        if i == j and a == b:
                            pool.append(of.ops.FermionOperator(str(2*b+1)+'^ '+str(2*a)+'^ '+str(2*i)+' '+str(2*j+1), 1))
                            v_pool.append(f"{i}{j}->{a}{b}")
                        elif i == j:
                            pool.append(of.ops.FermionOperator(str(2*b+1)+'^ '+str(2*a)+'^ '+str(2*i)+' '+str(2*j+1), 1/np.sqrt(2)) + of.ops.FermionOperator(str(2*a+1)+'^ '+str(2*b)+'^ '+str(2*i)+' '+str(2*j+1), 1/np.sqrt(2)))
                            v_pool.append(f"{i}{j}->{a}{b}")
                        elif a == b:
                            pool.append(of.ops.FermionOperator(str(2*b+1)+'^ '+str(2*a)+'^ '+str(2*i)+' '+str(2*j+1), 1/np.sqrt(2)) + of.ops.FermionOperator(str(2*b+1)+'^ '+str(2*a)+'^ '+str(2*j)+' '+str(2*i+1), 1/np.sqrt(2)))
                            v_pool.append(f"{i}{j}->{a}{b}")
                        else:
                            pool.append(of.ops.FermionOperator(str(2*b)+'^ '+str(2*a)+'^ '+str(2*i)+' '+str(2*j), 2/np.sqrt(12)) + of.ops.FermionOperator(str(2*b+1)+'^ '+str(2*a+1)+'^ '+str(2*i+1)+' '+str(2*j+1), 2/np.sqrt(12))+ of.ops.FermionOperator(str(2*b+1)+'^ '+str(2*a)+'^ '+str(2*i)+' '+str(2*j+1), 1/np.sqrt(12)) + of.ops.FermionOperator(str(2*a+1)+'^ '+str(2*b)+'^ '+str(2*j)+' '+str(2*i+1), 1/np.sqrt(12)) + of.ops.FermionOperator(str(2*b+1)+'^ '+str(2*a)+'^ '+str(2*j)+' '+str(2*i+1), -1/np.sqrt(12))+ of.ops.FermionOperator(str(2*a+1)+'^ '+str(2*b)+'^ '+str(2*i)+' '+str(2*j+1), -1/np.sqrt(12)) )
                            v_pool.append(f"{i}{j}->{a}{b} (Type 1)")
                            pool.append(of.ops.FermionOperator(str(2*b+1)+'^ '+str(2*a)+'^ '+str(2*i)+' '+str(2*j+1), 1/2)+ of.ops.FermionOperator(str(2*b+1)+'^ '+str(2*a)+'^ '+str(2*j)+' '+str(2*i+1), 1/2)+ of.ops.FermionOperator(str(2*a+1)+'^ '+str(2*b)+'^ '+str(2*i)+' '+str(2*j+1), 1/2)+ of.ops.FermionOperator(str(2*a+1)+'^ '+str(2*b)+'^ '+str(2*j)+' '+str(2*i+1), 1/2))
                            v_pool.append(f"{i}{j}->{a}{b} (Type 2)")
            
        #Adding normalization
        for i in range(0, len(pool)):
             op = copy.copy(pool[i])
             op -= of.hermitian_conjugated(op)
             op = of.normal_ordered(op)
             assert(op.many_body_order() > 0)

        jw_pool = [scipy.sparse.csr.csr_matrix(of.linalg.get_sparse_operator(i, n_qubits = N_qubits).real) for i in pool]
        self.pool = pool
        print("Operators in pool:")
        print(len(pool)) 
        return jw_pool, v_pool

    def uccs_then_d_pool(self, approach = 'vanilla'):
        N_qubits = self.N_qubits
        N_e = self.N_e
        pool = []
        v_pool = []
        if approach == 'vanilla':
            for i in range(0, N_e):
                for a in range(N_e, N_qubits):
                    if (i+a)%2 == 0:
                        pool.append(of.ops.FermionOperator(str(a)+'^ '+str(i), 1))
                        v_pool.append(f"{i}->{a}")
            for i in range(0, N_e):
                for a in range(N_e, N_qubits):
                    for j in range(i+1, N_e):
                        for b in range(a+1, N_qubits):
                            if i%2+j%2 == a%2+b%2:
                                pool.append(of.ops.FermionOperator(str(b)+'^ '+str(a)+'^ '+str(i)+' '+str(j), 1))
                                v_pool.append(f"{i},{j}->{a},{b}")

        for i in range(0, len(pool)):
             op = copy.copy(pool[i])
             op -= of.hermitian_conjugated(op)
             op = of.normal_ordered(op)
             assert(op.many_body_order() > 0)

        jw_pool = [scipy.sparse.csr.csr_matrix(of.linalg.get_sparse_operator(i, n_qubits = N_qubits).real) for i in pool]

        self.pool = pool
        print("Operators in pool:")
        print(len(pool)) 
        return jw_pool, v_pool
    
    def pair_pool(self):
        """UPCCGSD-based pool constructor
        Returns
        -------
        jw_pool, v_pool : list
            sparse matrices and verbal representations of operators respectively
        """ 
        N_qubits = self.N_qubits
        N_e = self.N_e
        pool = []
        v_pool = []
        for p in range(0, int(N_qubits/2)):
            for q in range(p+1, int(N_qubits/2)):
                 pool.append(of.ops.FermionOperator(str(2*q)+'^ '+str(2*p), 1)+of.ops.FermionOperator(str(2*q+1)+'^ '+str(2*p+1)))
                 v_pool.append(f"{2*p}->{2*q}")
                 pool.append(of.ops.FermionOperator(str(2*q+1)+'^ '+str(2*q)+'^ '+str(2*p)+' '+str(2*p+1), 1))
                 v_pool.append(f"{p},{p}->{q},{q}")
        #Normalized based on action on reference.
        for i in range(0, len(pool)):
             op = copy.copy(pool[i])
             op -= of.hermitian_conjugated(op)
             op = of.normal_ordered(op)
             assert(op.many_body_order() > 0)

        jw_pool = [scipy.sparse.csr.csr_matrix(of.linalg.get_sparse_operator(i, n_qubits = N_qubits).real) for i in pool]

        self.pool = pool
        print("Operators in pool:")
        print(len(pool)) 
        return jw_pool, v_pool

    def uccsd_pool(self, approach = 'vanilla'):
        """UCCSD-based pool constructor

        Parameters
        ----------
        approach : string
            'vanilla', 'spin_complement', or 'spin_adapt'

        Returns
        -------
        jw_pool, v_pool : list
            sparse matrices and verbal representations of operators respectively
        """ 
        N_qubits = self.N_qubits
        N_e = self.N_e
        pool = []
        v_pool = []
        if approach == 'vanilla':
            for i in range(0, N_e):
                for a in range(N_e, N_qubits):
                    if (i+a)%2 == 0:
                        pool.append(of.ops.FermionOperator(str(a)+'^ '+str(i), 1))
                        v_pool.append(f"{i}->{a}")
                    for j in range(i+1, N_e):
                        for b in range(a+1, N_qubits):
                            if i%2+j%2 == a%2+b%2:
                                pool.append(of.ops.FermionOperator(str(b)+'^ '+str(a)+'^ '+str(i)+' '+str(j), 1))
                                v_pool.append(f"{i},{j}->{a},{b}")
       
        elif approach == 'spin_complement':
           M = int(N_qubits/2)
           N = int(N_e/2)
           for i in range(0, N):
               for a in range(N, M):
                   pool.append(of.ops.FermionOperator(str(2*a)+'^ '+str(2*i), 1/np.sqrt(2))+of.ops.FermionOperator(str(2*a+1)+'^ '+str(2*i+1), 1/np.sqrt(2)))
                   v_pool.append(f"{i}->{a}")
                   for j in range(i, N):
                       for b in range(a, M):
                               if i != j and a != b:
                                   pool.append(of.ops.FermionOperator(str(2*b)+'^ '+str(2*a)+'^ '+str(2*i)+' '+str(2*j), 1/np.sqrt(2)) + of.ops.FermionOperator(str(2*b+1)+'^ '+str(2*a+1)+'^ '+str(2*i+1)+' '+str(2*j+1), 1/np.sqrt(2)))
                                   v_pool.append(f"{2*i}{2*j}->{2*a}{2*b}")                                               
                               pool.append(of.ops.FermionOperator(str(2*b+1)+'^ '+str(2*a)+'^ '+str(2*i)+' '+str(2*j+1), 1/np.sqrt(2)) + of.ops.FermionOperator(str(2*a+1)+'^ '+str(2*b)+'^ '+str(2*j)+' '+str(2*i+1), 1/np.sqrt(2)))
                               v_pool.append(f"{2*i}{2*j+1}->{2*a}{2*b+1}")
                               if i != j and a != b:
                                   pool.append(of.ops.FermionOperator(str(2*b+1)+'^ '+str(2*a)+'^ '+str(2*j)+' '+str(2*i+1), 1/np.sqrt(2)) + of.ops.FermionOperator(str(2*a+1)+'^ '+str(2*b)+'^ '+str(2*i)+' '+str(2*j+1), 1/np.sqrt(2)))
                                   v_pool.append(f"{2*j}{2*i+1}->{2*a}{2*b+1}")
                                    
        elif approach == 'spin_adapt':
           M = int(N_qubits/2)
           N = int(N_e/2)
           for i in range(0, N):
               for a in range(N, M):
                   pool.append(of.ops.FermionOperator(str(2*a)+'^ '+str(2*i), 1/np.sqrt(2))+of.ops.FermionOperator(str(2*a+1)+'^ '+str(2*i+1), 1/np.sqrt(2)))
                   v_pool.append(f"{i}->{a}")
                   for j in range(i, N):
                       for b in range(a, M):
                           if (i, j) != (a, b):
                               if i == j and a == b:
                                   pool.append(of.ops.FermionOperator(str(2*b+1)+'^ '+str(2*a)+'^ '+str(2*i)+' '+str(2*j+1), 1))
                                   v_pool.append(f"{i}{j}->{a}{b}")
                               elif i == j:
                                   pool.append(of.ops.FermionOperator(str(2*b+1)+'^ '+str(2*a)+'^ '+str(2*i)+' '+str(2*j+1), 1/np.sqrt(2)) + of.ops.FermionOperator(str(2*a+1)+'^ '+str(2*b)+'^ '+str(2*i)+' '+str(2*j+1), 1/np.sqrt(2)))
                                   v_pool.append(f"{i}{j}->{a}{b}")
                               elif a == b:
                                   pool.append(of.ops.FermionOperator(str(2*b+1)+'^ '+str(2*a)+'^ '+str(2*i)+' '+str(2*j+1), 1/np.sqrt(2)) + of.ops.FermionOperator(str(2*b+1)+'^ '+str(2*a)+'^ '+str(2*j)+' '+str(2*i+1), 1/np.sqrt(2)))
                                   v_pool.append(f"{i}{j}->{a}{b}")
                               else:
                                   pool.append(of.ops.FermionOperator(str(2*b)+'^ '+str(2*a)+'^ '+str(2*i)+' '+str(2*j), 2/np.sqrt(12)) + of.ops.FermionOperator(str(2*b+1)+'^ '+str(2*a+1)+'^ '+str(2*i+1)+' '+str(2*j+1), 2/np.sqrt(12))+ of.ops.FermionOperator(str(2*b+1)+'^ '+str(2*a)+'^ '+str(2*i)+' '+str(2*j+1), 1/np.sqrt(12)) + of.ops.FermionOperator(str(2*a+1)+'^ '+str(2*b)+'^ '+str(2*j)+' '+str(2*i+1), 1/np.sqrt(12)) + of.ops.FermionOperator(str(2*b+1)+'^ '+str(2*a)+'^ '+str(2*j)+' '+str(2*i+1), -1/np.sqrt(12))+ of.ops.FermionOperator(str(2*a+1)+'^ '+str(2*b)+'^ '+str(2*i)+' '+str(2*j+1), -1/np.sqrt(12)) )
                                   v_pool.append(f"{i}{j}->{a}{b} (Type 1)")
                                   pool.append(of.ops.FermionOperator(str(2*b+1)+'^ '+str(2*a)+'^ '+str(2*i)+' '+str(2*j+1), 1/2)+ of.ops.FermionOperator(str(2*b+1)+'^ '+str(2*a)+'^ '+str(2*j)+' '+str(2*i+1), 1/2)+ of.ops.FermionOperator(str(2*a+1)+'^ '+str(2*b)+'^ '+str(2*i)+' '+str(2*j+1), 1/2)+ of.ops.FermionOperator(str(2*a+1)+'^ '+str(2*b)+'^ '+str(2*j)+' '+str(2*i+1), 1/2))
                                   v_pool.append(f"{i}{j}->{a}{b} (Type 2)")
            
        #Normalized based on action on reference.
        for i in range(0, len(pool)):
             op = copy.copy(pool[i])
             op -= of.hermitian_conjugated(op)
             op = of.normal_ordered(op)
             assert(op.many_body_order() > 0)

        jw_pool = [scipy.sparse.csr.csr_matrix(of.linalg.get_sparse_operator(i, n_qubits = N_qubits).real) for i in pool]

        self.pool = pool
        print("Operators in pool:")
        print(len(pool)) 
        return jw_pool, v_pool

    def sc_uccsd_pool(self):
        N_qubits = self.N_qubits
        N_e = self.N_e
        pool = []
        v_pool = []
        M = int(N_qubits/2)
        N = int(N_e/2)
        for i in range(0, N):
            for a in range(N, M):
                pool.append(of.ops.FermionOperator(str(2*a)+'^ '+str(2*i), 1/np.sqrt(2))+of.ops.FermionOperator(str(2*a+1)+'^ '+str(2*i+1), 1/np.sqrt(2)))
                v_pool.append(f"{i}->{a}")
                for j in range(i, N):
                    for b in range(a, M):
                         if i == j and a == b:
                             pool.append(of.ops.FermionOperator(str(2*b+1)+'^ '+str(2*a)+'^ '+str(2*i)+' '+str(2*j+1), 1))
                             v_pool.append(f"{i}{j}->{a}{b}")
                         elif i == j:
                             pool.append(of.ops.FermionOperator(str(2*b+1)+'^ '+str(2*a)+'^ '+str(2*i)+' '+str(2*j+1), 1/np.sqrt(2)) + of.ops.FermionOperator(str(2*a+1)+'^ '+str(2*b)+'^ '+str(2*i)+' '+str(2*j+1), 1/np.sqrt(2)))
                             v_pool.append(f"{i}{j}->{a}{b}")
                         elif a == b:
                             pool.append(of.ops.FermionOperator(str(2*b+1)+'^ '+str(2*a)+'^ '+str(2*i)+' '+str(2*j+1), 1/np.sqrt(2)) + of.ops.FermionOperator(str(2*b+1)+'^ '+str(2*a)+'^ '+str(2*j)+' '+str(2*i+1), 1/np.sqrt(2)))
                             v_pool.append(f"{i}{j}->{a}{b}")
                         else:
                             pool.append(of.ops.FermionOperator(str(2*b+1)+'^ '+str(2*a)+'^ '+str(2*i)+' '+str(2*j+1), 1/np.sqrt(2)) + of.ops.FermionOperator(str(2*a+1)+'^ '+str(2*b)+'^ '+str(2*j)+' '+str(2*i+1), 1/np.sqrt(2)))
                             v_pool.append(f"{i}{j}\'->{a}{b}\'")
                             pool.append(of.ops.FermionOperator(str(2*a+1)+'^ '+str(2*b)+'^ '+str(2*i)+' '+str(2*j+1), 1/np.sqrt(2)) + of.ops.FermionOperator(str(2*b+1)+'^ '+str(2*a)+'^ '+str(2*j)+' '+str(2*i+1), 1/np.sqrt(2)))
                             v_pool.append(f"{i}{j}\'->{b}{a}\'")
                             pool.append(of.ops.FermionOperator(str(2*b)+'^ '+str(2*a)+'^ '+str(2*i)+' '+str(2*j), 1/np.sqrt(2)) + of.ops.FermionOperator(str(2*b+1)+'^ '+str(2*a+1)+'^ '+str(2*i+1)+' '+str(2*j+1), 1/np.sqrt(2)))
                             v_pool.append(f"{i}{j}->{a}{b}")


            
        #Adding normalization
        for i in range(0, len(pool)):
             op = copy.copy(pool[i])
             op -= of.hermitian_conjugated(op)
             op = of.normal_ordered(op)
             assert(op.many_body_order() > 0)
             coeff = 0
             for t in op.terms:                 
                 coeff_t = op.terms[t]
                 coeff += coeff_t * coeff_t
             op = op/np.sqrt(coeff)
             pool[i] = copy.copy(op)

        jw_pool = [scipy.sparse.csr.csr_matrix(of.linalg.get_sparse_operator(i, n_qubits = N_qubits).real) for i in pool]

        self.pool = pool
        print("Operators in pool:")
        print(len(pool)) 
        return jw_pool, v_pool

    def raw_uccsd_pool(self, spin_adapt = False):
        N_qubits = self.N_qubits
        N_e = self.N_e
        pool = []
        v_pool = []
        if spin_adapt == False:
            for i in range(0, N_e):
                for a in range(N_e, N_qubits):
                    if (i+a)%2 == 0:
                        pool.append(of.ops.FermionOperator(str(a)+'^ '+str(i), 1))
                        v_pool.append(f"{i}->{a}")
            for i in range(0, N_e):
                for a in range(N_e, N_qubits):
                    for j in range(i+1, N_e):
                        for b in range(a+1, N_qubits):
                            if i%2+j%2 == a%2+b%2:
                                pool.append(of.ops.FermionOperator(str(b)+'^ '+str(a)+'^ '+str(i)+' '+str(j), 1))
                                v_pool.append(f"{i},{j}->{a},{b}")
        elif spin_adapt == True:
           M = int(N_qubits/2)
           N = int(N_e/2)
           for i in range(0, N):
               for a in range(N, M):
                   pool.append(of.ops.FermionOperator(str(2*a)+'^ '+str(2*i), 1/np.sqrt(2))+of.ops.FermionOperator(str(2*a+1)+'^ '+str(2*i+1), 1/np.sqrt(2)))
                   v_pool.append(f"{i}->{a}")
                   for j in range(i, N):
                       for b in range(a, M):
                           if (i, j) != (a, b):

                               if i == j and a == b:
                                   pool.append(of.ops.FermionOperator(str(2*b+1)+'^ '+str(2*a)+'^ '+str(2*i)+' '+str(2*j+1), 1))

                                   v_pool.append(f"{i}{j}->{a}{b}")
                               elif i == j:
                                   pool.append(of.ops.FermionOperator(str(2*b+1)+'^ '+str(2*a)+'^ '+str(2*i)+' '+str(2*j+1), 1/np.sqrt(2)) + of.ops.FermionOperator(str(2*a+1)+'^ '+str(2*b)+'^ '+str(2*i)+' '+str(2*j+1), 1/np.sqrt(2)))
                                   v_pool.append(f"{i}{j}->{a}{b}")
                               elif a == b:
                                   pool.append(of.ops.FermionOperator(str(2*b+1)+'^ '+str(2*a)+'^ '+str(2*i)+' '+str(2*j+1), 1/np.sqrt(2)) + of.ops.FermionOperator(str(2*b+1)+'^ '+str(2*a)+'^ '+str(2*j)+' '+str(2*i+1), 1/np.sqrt(2)))
                                   v_pool.append(f"{i}{j}->{a}{b}")
                               else:
                                   pool.append(of.ops.FermionOperator(str(2*b)+'^ '+str(2*a)+'^ '+str(2*i)+' '+str(2*j), 2/np.sqrt(12)) + of.ops.FermionOperator(str(2*b+1)+'^ '+str(2*a+1)+'^ '+str(2*i+1)+' '+str(2*j+1), 2/np.sqrt(12))+ of.ops.FermionOperator(str(2*b+1)+'^ '+str(2*a)+'^ '+str(2*i)+' '+str(2*j+1), 1/np.sqrt(12)) + of.ops.FermionOperator(str(2*a+1)+'^ '+str(2*b)+'^ '+str(2*j)+' '+str(2*i+1), 1/np.sqrt(12)) + of.ops.FermionOperator(str(2*b+1)+'^ '+str(2*a)+'^ '+str(2*j)+' '+str(2*i+1), -1/np.sqrt(12))+ of.ops.FermionOperator(str(2*a+1)+'^ '+str(2*b)+'^ '+str(2*i)+' '+str(2*j+1), -1/np.sqrt(12)) )
                                   v_pool.append(f"{i}{j}->{a}{b} (Type 1)")
                                   pool.append(of.ops.FermionOperator(str(2*b+1)+'^ '+str(2*a)+'^ '+str(2*i)+' '+str(2*j+1), 1/2)+ of.ops.FermionOperator(str(2*b+1)+'^ '+str(2*a)+'^ '+str(2*j)+' '+str(2*i+1), 1/2)+ of.ops.FermionOperator(str(2*a+1)+'^ '+str(2*b)+'^ '+str(2*i)+' '+str(2*j+1), 1/2)+ of.ops.FermionOperator(str(2*a+1)+'^ '+str(2*b)+'^ '+str(2*j)+' '+str(2*i+1), 1/2))
                                   v_pool.append(f"{i}{j}->{a}{b} (Type 2)")
            
        #Adding normalization
        for i in range(0, len(pool)):
             op = copy.copy(pool[i])
             op -= of.hermitian_conjugated(op)
             op = of.normal_ordered(op)
             assert(op.many_body_order() > 0)

        jw_pool = [scipy.sparse.csr.csr_matrix(of.linalg.get_sparse_operator(i, n_qubits = N_qubits).real) for i in pool]

        self.pool = pool
        print("Operators in pool:")
        print(len(pool)) 
        return jw_pool, v_pool
 
    def qeb_pool(self):
        """
        n_orb is number of spatial orbitals assuming that spin orbitals are labelled
        0a,0b,1a,1b,2a,2b,3a,3b,....  -> 0,1,2,3,...
        """
        self.n_orb = int(self.N_qubits/2)
        self.n_spin_orb = self.N_qubits

        self.fermi = []
        v_pool = []
        real_vpool = []
        for p in range(0,self.n_orb):
            pa = 2*p
            pb = 2*p+1

            for q in range(p,self.n_orb):
                qa = 2*q
                qb = 2*q+1

                termA =  of.ops.FermionOperator(((pa,1),(qa,0)))
                termA -= of.hermitian_conjugated(termA)
                termA = of.normal_ordered(termA)
                if termA.many_body_order() > 0:
                    self.fermi.append(termA)
                    v_pool.append(f"{pa} -> {qa}")

                termA = of.ops.FermionOperator(((pb,1),(qb,0)))
                termA -= of.hermitian_conjugated(termA)
                termA = of.normal_ordered(termA)
                if termA.many_body_order() > 0:
                    self.fermi.append(termA)
                    v_pool.append(f"{pb} -> {qb}")

        pq = -1
        for p in range(0,self.n_orb):
            pa = 2*p
            pb = 2*p+1

            for q in range(p,self.n_orb):
                qa = 2*q
                qb = 2*q+1

                pq += 1

                rs = -1
                for r in range(0,self.n_orb):
                    ra = 2*r
                    rb = 2*r+1

                    for s in range(r,self.n_orb):
                        sa = 2*s
                        sb = 2*s+1

                        rs += 1

                        if(pq > rs):
                            continue

                        termA =  of.ops.FermionOperator(((ra,1),(pa,0),(sa,1),(qa,0)))
                        termA -= of.hermitian_conjugated(termA)
                        termA = of.normal_ordered(termA)
                        if termA.many_body_order() > 0:
                            self.fermi.append(termA)
                            v_pool.append(f"{pa} {qa} -> {ra} {sa}")

                        termA = of.ops.FermionOperator(((rb,1),(pb,0),(sb,1),(qb,0)))
                        termA -= of.hermitian_conjugated(termA)
                        termA = of.normal_ordered(termA)
                        if termA.many_body_order() > 0:
                            self.fermi.append(termA)
                            v_pool.append(f"{pb} {qb} -> {rb} {sb}")

                        termA = of.ops.FermionOperator(((ra,1),(pa,0),(sb,1),(qb,0)))
                        termA -= of.hermitian_conjugated(termA)
                        termA = of.normal_ordered(termA)
                        if termA.many_body_order() > 0:
                            self.fermi.append(termA)
                            v_pool.append(f"{pa} {qb} -> {ra} {sb}")

                        termA = of.ops.FermionOperator(((rb,1),(pb,0),(sa,1),(qa,0)))
                        termA -= of.hermitian_conjugated(termA)
                        termA = of.normal_ordered(termA)
                        if termA.many_body_order() > 0:
                            self.fermi.append(termA)
                            v_pool.append(f"{pb} {qa} -> {rb} {sa}")

                        termA = of.ops.FermionOperator(((ra,1),(pb,0),(sb,1),(qa,0)))
                        termA -= of.hermitian_conjugated(termA)
                        termA = of.normal_ordered(termA)
                        if termA.many_body_order() > 0:
                            self.fermi.append(termA)
                            v_pool.append(f"{pb} {qa} -> {ra} {sb}")

                        termA = of.ops.FermionOperator(((rb,1),(pa,0),(sa,1),(qb,0)))
                        termA -= of.hermitian_conjugated(termA)
                        termA = of.normal_ordered(termA)
                        if termA.many_body_order() > 0:
                            self.fermi.append(termA)
                            v_pool.append(f"{pa} {qb} -> {rb} {sa}")

        self.n_ops = len(self.fermi)

        n = self.n_spin_orb

        pool_vec = np.zeros((4 ** n,))

        self.fermi_ops = []
        opqubits=[]
        real_v_pool = []
        for j in range(0, len(self.fermi)):
            i = self.fermi[j]
            pauli = of.transforms.jordan_wigner(i)
            op = of.ops.QubitOperator()
            for line in pauli.terms:
                opqlist=[]
                coeff = pauli.terms[line]
                line = str(line)
                # print(line)
                Bin = np.zeros((2 * n,), dtype=int)
                X_pat_1 = re.compile("(\d{,2}), 'X'")
                X_1 = X_pat_1.findall(line)
                if X_1:
                    for i in X_1:
                        k = int(i)
                        Bin[n + k] = 1
                Y_pat_1 = re.compile("(\d{,2}), 'Y'")
                Y_1 = Y_pat_1.findall(line)
                if Y_1:
                    for i in Y_1:
                        k = int(i)
                        Bin[n + k] = 1
                        Bin[k] = 1
                Z_pat_1 = re.compile("(\d{,2}), 'Z'")
                Z_1 = Z_pat_1.findall(line)
                if Z_1:
                    for i in Z_1:
                        k = int(i)
                        Bin[k] = 1
                # print(Bin)
                index = int("".join(str(x) for x in Bin), 2)
                # print("index", index)

                pool_vec[index] = int(1)

                pauli_string = ''
                flip = []
                qstring=''
                for k in range(n):
                    if Bin[k] == 0:
                        if Bin[k + n] == 1:
                            pauli_string += 'X%d ' % k
                            flip.append(k)
                            opqlist.append(k)
                    if Bin[k] == 1:
                        if Bin[k + n] == 1:
                            pauli_string += 'Y%d ' % k
                            flip.append(k)
                            opqlist.append(k)
                flip.sort()
                opqlist.sort()
                opqlist = [str(x) for x in opqlist]
                qstring=' '.join(opqlist)
                z_string = list(range(flip[0] + 1,flip[1]))
                if len(flip) == 4:
                    for i in range(flip[2] + 1, flip[3]):
                        z_string.append(i)
                for i in z_string:
                    Bin[i] += 1
                    Bin[i] = Bin[i] % 2
                A = of.ops.QubitOperator(pauli_string, coeff)
                op += A 

            if qstring not in opqubits:
                opqubits.append(qstring)
                self.fermi_ops.append(op)
                real_vpool.append(v_pool[j])

        self.n_ops = len(self.fermi_ops)
        print(" Number of qubit excitation operators: ", self.n_ops)
        jw_pool = [scipy.sparse.csr.csr_matrix(of.linalg.get_sparse_operator(i, n_qubits = self.N_qubits).real) for i in self.fermi_ops]
        return jw_pool, real_vpool

    def vccsd_pool(self, spin_adapt = False):
        N_qubits = self.N_qubits
        N_e = self.N_e
        pool = []
        v_pool = []
        if spin_adapt == False:
            for i in range(0, N_e):
                for a in range(N_e, N_qubits):
                    if (i+a)%2 == 0:
                        pool.append(of.ops.FermionOperator(str(a)+'^ '+str(i), 1))
                        v_pool.append(f"{i}->{a}")
                    for j in range(i+1, N_e):
                        for b in range(a+1, N_qubits):
                            if i%2+j%2 == a%2+b%2:
                                pool.append(of.ops.FermionOperator(str(b)+'^ '+str(a)+'^ '+str(i)+' '+str(j), 1))
                                v_pool.append(f"{i},{j}->{a},{b}")
        elif spin_adapt == True:
           M = int(N_qubits/2)
           N = int(N_e/2)
           for i in range(0, N):
               for a in range(N, M):
                   pool.append(of.ops.FermionOperator(str(2*a)+'^ '+str(2*i), 1/np.sqrt(2))+of.ops.FermionOperator(str(2*a+1)+'^ '+str(2*i+1), 1/np.sqrt(2)))
                   v_pool.append(f"{i}->{a}")
                   for j in range(i, N):
                       for b in range(a, M):
                           if (i, j) != (a, b):

                               if i == j and a == b:
                                   pool.append(of.ops.FermionOperator(str(2*b+1)+'^ '+str(2*a)+'^ '+str(2*i)+' '+str(2*j+1), 1))

                                   v_pool.append(f"{i}{j}->{a}{b}")
                               elif i == j:
                                   pool.append(of.ops.FermionOperator(str(2*b+1)+'^ '+str(2*a)+'^ '+str(2*i)+' '+str(2*j+1), 1/np.sqrt(2)) + of.ops.FermionOperator(str(2*a+1)+'^ '+str(2*b)+'^ '+str(2*i)+' '+str(2*j+1), 1/np.sqrt(2)))
                                   v_pool.append(f"{i}{j}->{a}{b}")
                               elif a == b:
                                   pool.append(of.ops.FermionOperator(str(2*b+1)+'^ '+str(2*a)+'^ '+str(2*i)+' '+str(2*j+1), 1/np.sqrt(2)) + of.ops.FermionOperator(str(2*b+1)+'^ '+str(2*a)+'^ '+str(2*j)+' '+str(2*i+1), 1/np.sqrt(2)))
                                   v_pool.append(f"{i}{j}->{a}{b}")
                               else:
                                   pool.append(of.ops.FermionOperator(str(2*b)+'^ '+str(2*a)+'^ '+str(2*i)+' '+str(2*j), 2/np.sqrt(12)) + of.ops.FermionOperator(str(2*b+1)+'^ '+str(2*a+1)+'^ '+str(2*i+1)+' '+str(2*j+1), 2/np.sqrt(12))+ of.ops.FermionOperator(str(2*b+1)+'^ '+str(2*a)+'^ '+str(2*i)+' '+str(2*j+1), 1/np.sqrt(12)) + of.ops.FermionOperator(str(2*a+1)+'^ '+str(2*b)+'^ '+str(2*j)+' '+str(2*i+1), 1/np.sqrt(12)) + of.ops.FermionOperator(str(2*b+1)+'^ '+str(2*a)+'^ '+str(2*j)+' '+str(2*i+1), -1/np.sqrt(12))+ of.ops.FermionOperator(str(2*a+1)+'^ '+str(2*b)+'^ '+str(2*i)+' '+str(2*j+1), -1/np.sqrt(12)) )
                                   v_pool.append(f"{i}{j}->{a}{b} (Type 1)")
                                   pool.append(of.ops.FermionOperator(str(2*b+1)+'^ '+str(2*a)+'^ '+str(2*i)+' '+str(2*j+1), 1/2)+ of.ops.FermionOperator(str(2*b+1)+'^ '+str(2*a)+'^ '+str(2*j)+' '+str(2*i+1), 1/2)+ of.ops.FermionOperator(str(2*a+1)+'^ '+str(2*b)+'^ '+str(2*i)+' '+str(2*j+1), 1/2)+ of.ops.FermionOperator(str(2*a+1)+'^ '+str(2*b)+'^ '+str(2*j)+' '+str(2*i+1), 1/2))
                                   v_pool.append(f"{i}{j}->{a}{b} (Type 2)")
            


        jw_pool = [scipy.sparse.csr.csr_matrix(of.linalg.get_sparse_operator(i, n_qubits = N_qubits).real) for i in pool]

        self.pool = pool
        print("Operators in pool:")

        return jw_pool, v_pool



    def uccgsd_pool(self, spin_adapt = False):
        """UCCGSD-based pool constructor

        Parameters
        ----------
        spin_adapt : Bool
            Do spin-adaptation of the pool?

        Returns
        -------
        jw_pool, v_pool : list
            sparse matrices and verbal representations of operators respectively
        """ 
        N_qubits = self.N_qubits
        N_e = self.N_e
        pool = []
        v_pool = []
        if spin_adapt == False:
            for i in range(0, N_qubits):
                for a in range(i+1, N_qubits):
                    if (i+a)%2 == 0:
                        pool.append(of.ops.FermionOperator(str(a)+'^ '+str(i), 1))
                        v_pool.append(f"{i}->{a}")
                    for j in range(i+1, N_qubits):
                        for b in range(a+1, N_qubits):
                            if i%2+j%2 == a%2+b%2:
                                pool.append(of.ops.FermionOperator(str(b)+'^ '+str(a)+'^ '+str(i)+' '+str(j), 1))
                                v_pool.append(f"{i}{j}->{a}{b}")


        elif spin_adapt == True:
           M = int(N_qubits/2)
           N = int(N_e/2)
           for p in range(0, M):
               pa = 2*p
               pb = 2*p+1

               for q in range(p, M):
                    qa = 2*q
                    qb = 2*q+1
                    termA =  of.ops.FermionOperator(((pa,1),(qa,0)))
                    termA += of.ops.FermionOperator(((pb,1),(qb,0)))

                    if of.normal_ordered(termA-of.hermitian_conjugated(termA)).many_body_order() > 0:
                        pool.append(termA)
                        v_pool.append(f"{p}->{q}")

           pq = -1
           for p in range(0,M):
                pa = 2*p
                pb = 2*p+1
    
                for q in range(p,M):
                    qa = 2*q
                    qb = 2*q+1
    
                    pq += 1
    
                    rs = -1
                    for r in range(0,M):
                        ra = 2*r
                        rb = 2*r+1
    
                        for s in range(r,M):
                            
                            sa = 2*s
                            sb = 2*s+1
                       
                            rs += 1
    
                            if(pq > rs):
                                continue
    
                            termA =  of.ops.FermionOperator(((ra,1),(pa,0),(sa,1),(qa,0)), 2/np.sqrt(12))
                            termA += of.ops.FermionOperator(((rb,1),(pb,0),(sb,1),(qb,0)), 2/np.sqrt(12))
                            termA += of.ops.FermionOperator(((ra,1),(pa,0),(sb,1),(qb,0)), 1/np.sqrt(12))
                            termA += of.ops.FermionOperator(((rb,1),(pb,0),(sa,1),(qa,0)), 1/np.sqrt(12))
                            termA += of.ops.FermionOperator(((ra,1),(pb,0),(sb,1),(qa,0)), 1/np.sqrt(12))
                            termA += of.ops.FermionOperator(((rb,1),(pa,0),(sa,1),(qb,0)), 1/np.sqrt(12))
    
                            termB =  of.ops.FermionOperator(((ra,1),(pa,0),(sb,1),(qb,0)),  1/2.0)
                            termB += of.ops.FermionOperator(((rb,1),(pb,0),(sa,1),(qa,0)),  1/2.0)
                            termB += of.ops.FermionOperator(((ra,1),(pb,0),(sb,1),(qa,0)), -1/2.0)
                            termB += of.ops.FermionOperator(((rb,1),(pa,0),(sa,1),(qb,0)), -1/2.0)

                            if of.normal_ordered(termA-of.hermitian_conjugated(termA)).many_body_order() > 0:
                                pool.append(termA)
                                v_pool.append(f"{p}{q}->{r}{s} (Type 1)")
    
                            if of.normal_ordered(termB-of.hermitian_conjugated(termB)).many_body_order() > 0:
                                pool.append(termB)
                                v_pool.append(f"{p}{q}->{r}{s} (Type 2)")
    



        #Adding normalization
        for i in range(0, len(pool)):
             op = copy.copy(pool[i])
             op -= of.hermitian_conjugated(op)
             op = of.normal_ordered(op)
             try: 
                 assert(op.many_body_order() > 0)
             except:
                 print(f"{v_pool[i]} has order 0.")
             coeff = 0
             for t in op.terms:                 
                 coeff_t = op.terms[t]
                 coeff += coeff_t * coeff_t
             op = op/np.sqrt(coeff)
             pool[i] = copy.copy(op)

        jw_pool = [scipy.sparse.csr.csr_matrix(of.linalg.get_sparse_operator(i, n_qubits = N_qubits).real) for i in pool]

        self.pool = pool
        print("Operators in pool:")
        print(len(pool)) 
        return jw_pool, v_pool


    def uccgs_pool(self, spin_adapt = True):
        N_qubits = self.N_qubits
        N_e = self.N_e
        pool = []
        v_pool = []
        if spin_adapt == False:
            for i in range(0, N_qubits):
                for a in range(i+1, N_qubits):
                    if (i+a)%2 == 0:
                        pool.append(of.ops.FermionOperator(str(a)+'^ '+str(i), 1))

        elif spin_adapt == True:
           M = int(N_qubits/2)
           N = int(N_e/2)
           for p in range(0, M):
               pa = 2*p
               pb = 2*p+1

               for q in range(p, M):
                    qa = 2*q
                    qb = 2*q+1
                    termA =  of.ops.FermionOperator(((pa,1),(qa,0)))
                    termA += of.ops.FermionOperator(((pb,1),(qb,0)))

                    if of.normal_ordered(termA-of.hermitian_conjugated(termA)).many_body_order() > 0:
                        pool.append(termA)
                        v_pool.append(f"{p}->{q}")

        print("UCCGS Operators:")
        for v in v_pool: 
            print(v)
        #Adding normalization
        for i in range(0, len(pool)):
             op = copy.copy(pool[i])
             op -= of.hermitian_conjugated(op)
             op = of.normal_ordered(op)
             try: 
                 assert(op.many_body_order() > 0)
             except:
                 print(f"{v_pool[i]} has order 0.")
             coeff = 0
             for t in op.terms:                 
                 coeff_t = op.terms[t]
                 coeff += coeff_t * coeff_t
             op = op/np.sqrt(coeff)
             pool[i] = copy.copy(op)

        jw_pool = [scipy.sparse.csr.csr_matrix(of.linalg.get_sparse_operator(i, n_qubits = N_qubits).real) for i in pool]

        self.pool = pool
        print("Operators in pool:")
        print(len(pool)) 
        return jw_pool, v_pool

    def uccd_pool(self, spin_adapt = True):
        N_qubits = self.N_qubits
        N_e = self.N_e
        pool = []
        v_pool = []
        if spin_adapt == False:
            for i in range(0, N_e):
                for a in range(N_e, N_qubits):
                    for j in range(i+1, N_e):
                        for b in range(a+1, N_qubits):
                            if i%2+j%2 == a%2+b%2:
                                pool.append(of.ops.FermionOperator(str(b)+'^ '+str(a)+'^ '+str(i)+' '+str(j), 1))
              
        elif spin_adapt == True:
           M = int(N_qubits/2)
           N = int(N_e/2)
           for i in range(0, N):
               for a in range(N, M):
                   for j in range(i, N):
                       for b in range(a, M):
                           if (i, j) != (a, b):
                               if i == j and a == b:
                                   pool.append(of.ops.FermionOperator(str(2*b+1)+'^ '+str(2*a)+'^ '+str(2*i)+' '+str(2*j+1), 1))
                                   v_pool.append(f"{i}{j}->{a}{b}")
                               elif i == j:
                                   pool.append(of.ops.FermionOperator(str(2*b+1)+'^ '+str(2*a)+'^ '+str(2*i)+' '+str(2*j+1), 1/np.sqrt(2)) + of.ops.FermionOperator(str(2*a+1)+'^ '+str(2*b)+'^ '+str(2*i)+' '+str(2*j+1), 1/np.sqrt(2)))
                                   v_pool.append(f"{i}{j}->{a}{b}")
                               elif a == b:
                                   pool.append(of.ops.FermionOperator(str(2*b+1)+'^ '+str(2*a)+'^ '+str(2*i)+' '+str(2*j+1), 1/np.sqrt(2)) + of.ops.FermionOperator(str(2*b+1)+'^ '+str(2*a)+'^ '+str(2*j)+' '+str(2*i+1), 1/np.sqrt(2)))
                                   v_pool.append(f"{i}{j}->{a}{b}")
                               else:
                                   pool.append(of.ops.FermionOperator(str(2*b)+'^ '+str(2*a)+'^ '+str(2*i)+' '+str(2*j), 2/np.sqrt(12)) + of.ops.FermionOperator(str(2*b+1)+'^ '+str(2*a+1)+'^ '+str(2*i+1)+' '+str(2*j+1), 2/np.sqrt(12))+ of.ops.FermionOperator(str(2*b+1)+'^ '+str(2*a)+'^ '+str(2*i)+' '+str(2*j+1), 1/np.sqrt(12)) + of.ops.FermionOperator(str(2*a+1)+'^ '+str(2*b)+'^ '+str(2*j)+' '+str(2*i+1), 1/np.sqrt(12)) + of.ops.FermionOperator(str(2*b+1)+'^ '+str(2*a)+'^ '+str(2*j)+' '+str(2*i+1), -1/np.sqrt(12))+ of.ops.FermionOperator(str(2*a+1)+'^ '+str(2*b)+'^ '+str(2*i)+' '+str(2*j+1), -1/np.sqrt(12)) )
                                   v_pool.append(f"{i}{j}->{a}{b} (Type 1)")
                                   pool.append(of.ops.FermionOperator(str(2*b+1)+'^ '+str(2*a)+'^ '+str(2*i)+' '+str(2*j+1), 1/2)+ of.ops.FermionOperator(str(2*b+1)+'^ '+str(2*a)+'^ '+str(2*j)+' '+str(2*i+1), 1/2)+ of.ops.FermionOperator(str(2*a+1)+'^ '+str(2*b)+'^ '+str(2*i)+' '+str(2*j+1), 1/2)+ of.ops.FermionOperator(str(2*a+1)+'^ '+str(2*b)+'^ '+str(2*j)+' '+str(2*i+1), 1/2))
                                   v_pool.append(f"{i}{j}->{a}{b} (Type 2)")
            
        #Adding normalization
        for i in range(0, len(pool)):
             op = copy.copy(pool[i])
             op -= of.hermitian_conjugated(op)
             op = of.normal_ordered(op)
             assert(op.many_body_order() > 0)
             coeff = 0
             for t in op.terms:                 
                 coeff_t = op.terms[t]
                 coeff += coeff_t * coeff_t
             op = op/np.sqrt(coeff)
             pool[i] = copy.copy(op)

        jw_pool = [scipy.sparse.csr.csr_matrix(of.linalg.get_sparse_operator(i, n_qubits = N_qubits).real) for i in pool]

        self.pool = pool
        print("Operators in pool:")
        print(len(pool)) 
        print("UCCD Operators:")
        for v in v_pool: 
            print(v)
        return jw_pool, v_pool
