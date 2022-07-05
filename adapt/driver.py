import os
os.environ["OPENBLAS_NUM_THREADS"] = '1'
import numpy as np
import adapt.system_methods as sm
import adapt.computational_tools as ct
from opt_einsum import contract
import openfermion as of
import scipy
import copy
import time
import math
import git

import random
from multiprocessing import Pool
#Globals
Eh = 627.5094740631

class Xiphos:
    """Class representing an individual XIPHOS calculation"""
    def __init__(self, H, ref, system, pool, v_pool, H_adapt = None, H_vqe = None, sym_ops = None):

        """Initialize a XIPHOS Solver Object.
        Parameters
        ----------
        H, ref : scipy sparse matrix
            Hamiltonian and reference wfn
        system : system object
            System object
        pool, v_pool : list
            Lists containing the actual operators (sparse matrices) and their verbal representations respectively
        H_adapt, H_vqe : scipy sparce matrix
            Hamiltonians to use for operator addition and VQE respectively, if different than H  
        sym_ops : dictionary
            Dictionary of string/sparse matrix pairs to check at each step of the calculation

        Returns
        -------
        None
        """
          
        self.H = H
        self.ref = ref
        self.system = system
        self.pool = pool
        self.v_pool = v_pool
        self.vqe_iteration = 0
        self.e_dict = {}
        self.grad_dict = {}
        self.state_dict = {}
 
        if H_adapt is None:
            self.H_adapt = self.H
        else:
            self.H_adapt = H_adapt

        if H_vqe is None:
            self.H_vqe = self.H
        else:
            self.H_vqe = H_vqe

      
        if sym_ops is None:
            self.sym_ops = {'H': H}
        else:
            self.sym_ops = sym_ops


        if os.path.isdir(system):
            self.restart = True
        else:
            os.mkdir(system)
            self.restart = False

        self.log_file = f"{system}/log.dat"

        if self.restart == False:
            print("Starting a new calculation.\n")
        else:  
            print("Restarting the calculation.\n")
        print("---------------------------")

         
        #We do an exact diagonalization to check for degeneracies/symmetry issues
        print("\nReference information:")
        self.ref_syms = {}
        for key in self.sym_ops.keys():
            val = (ref.T@(self.sym_ops[key]@ref))[0,0]
            print(f"{key: >6}: {val:20.16f}")
            self.ref_syms[key] = val
        self.hf = self.ref_syms['H']        
        print("\nED information:") 
        #w, v = scipy.sparse.linalg.eigsh(H, k = min(H.shape[0]-1,10), which = "SA")
        w, v = np.linalg.eigh(H.todense())
        k = min(H.shape[0]-1,10)
        self.ed_energies = w[:k]
        self.ed_wfns = v[:,:k]

        self.ed_syms = []
        for i in range(0, len(self.ed_energies)):
            print(f"ED Solution {i+1}:")
            ed_dict = {}
            for key in self.sym_ops.keys():
                val = np.asscalar(v[:,i].T@(self.sym_ops[key]@v[:,i])).real
                print(f"{key: >6}: {val:20.16f}")
                ed_dict[key] = copy.copy(val)
            self.ed_syms.append(copy.copy(ed_dict))

        for key in self.sym_ops.keys():
            if key != "H" and abs(self.ed_syms[0][key] - self.ref_syms[key]) > 1e-8:
                print(f"\nWARNING: <{key}> symmetry of reference inconsistent with ED solution.")
        if abs(self.ed_syms[0]["H"] - self.ed_syms[1]["H"]) < (1/Eh):
            print(f"\nWARNING:  Lowest two ED solutions may be quasi-degenerate.")

    def rebuild_ansatz(self, A):
        params = []
        ansatz = [] 
        #A is the number of operators in your ansatz that you know.
        os.system(f"grep -A{A} ansatz {self.log_file} > {self.system}/temp.dat")
        os.system(f"tail -n {A} {self.system}/temp.dat > {self.system}/temp2.dat")
        f = open(f"{self.system}/temp2.dat", "r")
        ansatz = []
        params = []
        for line in f.readlines():
            line = line.split()
            param = line[1]
            if len(line) == 5:
                op = line[2] + " " + line[3] + " " + line[4]
            else:
                op = line[2]
            #May or may not rebuild in correct order - double check
            ansatz = ansatz + [self.v_pool.index(op)]
            params = params + [float(param)]
        return params, ansatz
    
    def ucc_E(self, params, ansatz):
        G = params[0]*self.pool[ansatz[0]]
        for i in range(1, len(ansatz)):
            G += params[i]*self.pool[ansatz[i]] 
        state = scipy.sparse.linalg.expm_multiply(G, self.ref)
        E = ((state.T)@self.H@state).todense()[0,0].real
        return E



    def comm(self, A, B):
        return A@B - B@A

    def ucc_grad_zero(self, ansatz):
        grad = []
        for i in range(0, len(ansatz)):
            g = ((self.ref.T)@(self.comm(self.H, self.pool[ansatz[i]]))@self.ref).todense()[0,0]
            grad.append(g)
        return np.array(grad)
   
    def ucc_hess_zero(self, ansatz):
        hess = np.zeros((len(ansatz), len(ansatz)))
        for i in range(0, len(ansatz)):
            for j in range(0, len(ansatz)):
                    hess[i,j] += .5*((self.ref.T)@(self.comm(self.comm(self.H, self.pool[ansatz[i]]), self.pool[ansatz[j]])@self.ref)).todense()[0,0]
                    hess[j,i] += .5*((self.ref.T)@(self.comm(self.comm(self.H, self.pool[ansatz[i]]), self.pool[ansatz[j]])@self.ref)).todense()[0,0]
        return hess
    

    def ucc_diag_jerk_zero(self, ansatz):
        jerk = []
        for i in range(0, len(ansatz)):
            j = ((self.ref.T)@(self.comm(self.comm(self.comm(self.H, self.pool[ansatz[i]]), self.pool[ansatz[i]]), self.pool[ansatz[i]]))@self.ref).todense()[0,0]
            jerk.append(j)
        jmat = np.zeros((len(ansatz), len(ansatz), len(ansatz)))
        for i in range(0, len(jerk)):
            jmat[i,i,i] = jerk[i]
        return jmat

    def cubic_energy(self, x, grad, hess, jerk):
        return grad.T@x + .5*x.T@(hess@x) + (1/6)*contract('iii,i,i,i', jerk, x, x, x)

    def ucc_inf_d_E(self, params, ansatz, E0, grad, hess):
        E = E0 + grad.T@params + .5*params.T@hess@params
        for i in range(0, len(ansatz)):
            E += self.ucc_E(np.array([params[i]]), [ansatz[i]])
            E -= params[i]*grad[i]
            E -= .5*params[i]*params[i]*hess[i,i] 
            E -= E0
        return E
    
    def tucc_inf_d_E(self, params, ansatz, E0, grad, hess):
        E = E0 + grad.T@params + .5*params.T@hess@params
        for i in range(0, len(ansatz)):
            E += t_ucc_E(np.array([params[i]]), [ansatz[i]], self.H_vqe, self.pool, self.ref)
            E -= params[i]*grad[i]
            E -= .5*params[i]*params[i]*hess[i,i] 
            E -= E0
        return E

    def H_eff_analysis(self, params, ansatz):
        H_eff = copy.copy(self.H)
        for i in reversed(range(0, len(params))):
            U = scipy.sparse.linalg.expm(params[i]*self.pool[ansatz[i]])
            H_eff = ((U.T)@H@U).todense()
        E = ((self.ref.T)@H_eff@self.ref)
        print("Analysis of H_eff:")
        print(f"<0|H_eff|0> = {E}")
        w, v = np.linalg.eigh(H_eff)
        for sv in w:
            spec_string += f"{sv},"
        print(f"Eigenvalues of H_eff:")
        print(spec_string)
   

         
    def param_scan(self, params, ansatz, ref, a, b, save_file = "params.csv", gridpoints = 100):
        #a and b are the two indices to scan over.
        arr = np.zeros((gridpoints*gridpoints, 3))
        for i in range(0, gridpoints):
            for j in range(0, gridpoints):
                test_params = copy.copy(params)
                test_params[a] = i*2*math.pi/gridpoints               
                test_params[b] = j*2*math.pi/gridpoints
                arr[i*gridpoints+j,0] = test_params[a]
                arr[i*gridpoints+j,1] = test_params[b]
                arr[i*gridpoints+j,2] = t_ucc_E(test_params, ansatz, self.H_vqe, self.pool, self.ref)
        np.savetxt(save_file, arr, delimiter = ",")

    def two_vec_interpolate(self, theta_a, theta_b, ansatz):
        print("HF?")
        E = t_ucc_E(0*theta_a, ansatz, self.H_vqe, self.pool, self.ref)
        print(E)
        print("alpha,E")
        for i in range(0,101):
            alpha = i*.01
            theta = ((1-alpha) * theta_a) + (alpha * theta_b)
            E = t_ucc_E(theta, ansatz, self.H_vqe, self.pool, self.ref)
            print(f"{alpha},{E}", flush = True)
    
    def grad_variance(self, params, ansatz, ref, shots = 10, r = 2*math.pi, seed_base = 0):
        #Compute variance of gradient norm
        if r == 0:
            E = t_ucc_E(params, ansatz, self.H_vqe, self.pool, ref)
            print(f"Origin energy: {E}", flush = True)
        seeds = []
        grads = []
        param_list = []
        for i in range(0, shots):
            seed = i+seed_base
            seeds.append(seed)
            np.random.seed(seed)
            param_list.append(params + r*(2*np.random.rand(len(params))-1))
            #grads.append(t_ucc_grad(param_list[-1], ansatz, self.H_vqe, self.pool, ref))
         

        iterable = [*zip(param_list, [ansatz for i in range(0, len(param_list))], [self.H_vqe for j in range(0, len(param_list))], [self.pool for j in range(0, len(param_list))], [ref for j in range(0, len(param_list))])]
        with Pool(126) as p:
            grads = p.starmap(t_ucc_grad, iterable = iterable)
        with Pool(126) as p:
            energies = p.starmap(t_ucc_E, iterable = iterable)
        
        gnorms = [np.linalg.norm(np.array(g)) for g in grads]
        var = np.var(gnorms)
        #for i in range(0, len(param_list)):
        #    print(f"Seed: {seeds[i]}")
        #    print(f"Params:\n{param_list[i]}")
        #    print(f"Gradient:\n{grads[i]}")
        #print(f"Gradient Norm Variance: {var}", flush = True)

        for i in energies:
            print(str(r) + " " + str(i), flush = True)
        return var

    def pretend_adapt(self, params, ansatz, ref, order, guesses = 0):
        #use preordained operator sequence
        state = t_ucc_state(params, ansatz, self.pool, self.ref)
        iteration = len(ansatz)
        print(f"\nADAPT Iteration {iteration}")
        print("Performing (Pretend) ADAPT with Predetermined Operators:")
        E = (state.T@(self.H@state))[0,0] 
        Done = False
        for op in reversed(order):
            gradient = 2*np.array([((state.T@(self.H_adapt@(op2@state)))[0,0]) for op2 in self.pool]).real
            gnorm = np.linalg.norm(gradient)

                 
            E = (state.T@(self.H@state))[0,0].real 
            error = E - self.ed_energies[0]
            fid = ((self.ed_wfns[:,0].T)@state)[0,0].real**2
            print(f"\nBest Initialization Information:")
            print(f"Operator/ Expectation Value/ Error")

            for key in self.sym_ops.keys():
                val = ((state.T)@(self.sym_ops[key]@state))[0,0].real
                err = val - self.ed_syms[0][key]
                print(f"{key:<6}:      {val:20.16f}      {err:20.16f}")
                
            print(f"Next operator to be added: {self.v_pool[op]}")
            print(f"Operator multiplicity {1+ansatz.count(op)}.")                
            print(f"Associated gradient:       {gradient[op]:20.16f}")
            print(f"Gradient norm:             {gnorm:20.16f}")
            print(f"Fidelity to ED:            {fid:20.16f}")
            print(f"Current ansatz:")
            for i in range(0, len(ansatz)):
                print(f"{i} {params[i]} {self.v_pool[ansatz[i]]}") 
            print("|0>")  

            iteration += 1
            print(f"\nADAPT Iteration {iteration}")

            params = np.array([0] + list(params))
            ansatz = [op] + ansatz
            H_vqe = copy.copy(self.H_vqe)
            pool = copy.copy(self.pool)
            ref = copy.copy(self.ref)
            params = multi_vqe(params, ansatz, H_vqe, pool, ref, self, guesses = guesses)  

            state = t_ucc_state(params, ansatz, self.pool, self.ref)
            np.save(f"{self.system}/params", params)
            np.save(f"{self.system}/ops", ansatz)

            
        print(f"\nConverged ADAPT energy:    {E:20.16f}")            
        print(f"\nConverged ADAPT error:     {error:20.16f}")            
        print(f"\nConverged ADAPT gnorm:     {gnorm:20.16f}")            
        print(f"\nConverged ADAPT fidelity:  {fid:20.16f}")            
        print("\n---------------------------\n")
        print("\"Adapt.\" - Bear Grylls\n")
        print("\"ADAPT.\" - Harper \"Grimsley Bear\" Grimsley\n")
        repo = git.Repo(search_parent_directories=True)
        sha = repo.head.object.hexsha
        print(f"Git revision:\ngithub.com/hrgrimsl/adapt/commit/{sha}")


    def random_adapt(self, params, ansatz, ref, gtol = None, Etol = None, max_depth = None, criteria = 'grad', guesses = 0):
        #Random operator adapt
        state = t_ucc_state(params, ansatz, self.pool, self.ref)
        iteration = len(ansatz)
        print(f"\nADAPT Iteration {iteration}")
        print("Performing ADAPT:")
        E = (state.T@(self.H@state))[0,0] 
        Done = False
        while Done == False:

            gradient = 2*np.array([((state.T@(self.H_adapt@(op@state)))[0,0]) for op in self.pool]).real
            gnorm = np.linalg.norm(gradient)
            if criteria == 'grad':
                idx = np.argsort(abs(gradient))
            random.seed(len(ansatz)) 
            random.shuffle(idx)     
            E = (state.T@(self.H@state))[0,0].real 
            error = E - self.ed_energies[0]
            fid = ((self.ed_wfns[:,0].T)@state)[0,0].real**2
            print(f"\nBest Initialization Information:")
            print(f"Operator/ Expectation Value/ Error")

            for key in self.sym_ops.keys():
                val = ((state.T)@(self.sym_ops[key]@state))[0,0].real
                err = val - self.ed_syms[0][key]
                print(f"{key:<6}:      {val:20.16f}      {err:20.16f}")
                
            print(f"Next operator to be added: {self.v_pool[idx[-1]]}")
            print(f"Operator multiplicity {1+ansatz.count(idx[-1])}.")                
            print(f"Associated gradient:       {gradient[idx[-1]]:20.16f}")
            print(f"Gradient norm:             {gnorm:20.16f}")
            print(f"Fidelity to ED:            {fid:20.16f}")
            print(f"Current ansatz:")
            for i in range(0, len(ansatz)):
                print(f"{i} {params[i]} {self.v_pool[ansatz[i]]}") 
            print("|0>")  
            if gtol is not None and gnorm < gtol:
                Done = True
                print(f"\nADAPT finished.  (Gradient norm acceptable.)")
                continue
            if max_depth is not None and iteration+1 > max_depth:
                Done = True
                print(f"\nADAPT finished.  (Max depth reached.)")
                continue
            if Etol is not None and error < Etol:
                Done = True
                print(f"\nADAPT finished.  (Error acceptable.)")
                continue
            iteration += 1
            print(f"\nADAPT Iteration {iteration}")

            params = np.array([0] + list(params))
            ansatz = [idx[-1]] + ansatz
            H_vqe = copy.copy(self.H_vqe)
            pool = copy.copy(self.pool)
            ref = copy.copy(self.ref)
            params = multi_vqe(params, ansatz, H_vqe, pool, ref, self, guesses = guesses)  

            state = t_ucc_state(params, ansatz, self.pool, self.ref)
            np.save(f"{self.system}/params", params)
            np.save(f"{self.system}/ops", ansatz)
        self.e_dict = {}
        self.grad_dict = {}
        self.state_dict = {}
            
        print(f"\nConverged ADAPT energy:    {E:20.16f}")            
        print(f"\nConverged ADAPT error:     {error:20.16f}")            
        print(f"\nConverged ADAPT gnorm:     {gnorm:20.16f}")            
        print(f"\nConverged ADAPT fidelity:  {fid:20.16f}")            
        print("\n---------------------------\n")
        print("\"Adapt.\" - Bear Grylls\n")
        print("\"ADAPT.\" - Harper \"Grimsley Bear\" Grimsley\n")
        repo = git.Repo(search_parent_directories=True)
        sha = repo.head.object.hexsha
        print(f"Git revision:\ngithub.com/hrgrimsl/fixed_adapt/commit/{sha}")

    def breadapt(self, params, ansatz, ref, gtol = None, Etol = None, max_depth = None, guesses = 0, n = 1, hf = True, threads = 1, seed = 0, criteria = 'grad'):
        """Run an ADAPT^N Calculation
        Parameters
        ----------
        params, ansatz : list
            Lists of parameters and operator indices to use as the initial ansatz and parameters
            Only recommend using non-empty lists for N = 1
        ref : scipy sparse matrix
            Reference matrix
        gtol, Etol : float
            gradient norm and energy thresholds 
        max_depth : int
            Max number of operators to use
        guesses : int
            Number of random guesses to try at each step
        hf : bool
            Whether to try the HF (all zeros) initialization at each step
        threads : int
            Number of threads to use for multiple BFGS threads
 
        Returns
        -------
        error : float
             The error from exact diagonalization
        """
        bre_ansatz = copy.copy(ansatz)
        bre_params = np.array(params)
        np.random.seed(seed = seed)
        state = t_ucc_state(bre_params, bre_ansatz, self.pool, self.ref)
        iteration = len(bre_ansatz)
        print(f"\nADAPT Iteration {iteration}")
        print("Performing ADAPT:")
        E = (state.T@(self.H@state))[0,0] 
        Done = False
        while Done == False:
            if criteria == 'grad':
                gradient = 2*np.array([((state.T@(self.H_adapt@(op@state)))[0,0]) for op in self.pool]).real
                gnorm = np.linalg.norm(gradient)
                idx = np.argsort(abs(gradient)) 
            elif criteria == 'random':
                gradient = [float('NaN') for i in range(0, len(self.pool))] 
                gnorm = float('NaN')
                idx = [i for i in range(0, len(self.pool))] 
                random.shuffle(idx)    
            E = (state.T@(self.H@state))[0,0].real 
            error = E - self.ed_energies[0]
            fid = ((self.ed_wfns[:,0].T)@state)[0,0].real**2
            print(f"\nBest Initialization Information:")
            print(f"Operator/ Expectation Value/ Error")

            for key in self.sym_ops.keys():
                val = ((state.T)@(self.sym_ops[key]@state))[0,0].real
                err = val - self.ed_syms[0][key]
                print(f"{key:<6}:      {val:20.16f}      {err:20.16f}")
                
            print(f"Next operator to be added: {self.v_pool[idx[-1]]}")
            print(f"Operator multiplicity {1+ansatz.count(idx[-1])}.")                
            print(f"Associated gradient:       {gradient[idx[-1]]:20.16f}")
            print(f"Gradient norm:             {gnorm:20.16f}")
            print(f"Fidelity to ED:            {fid:20.16f}")
            print(f"Current ansatz:")
            for i in range(0, len(bre_ansatz)):
                print(f"{i} {bre_params[i]} {self.v_pool[bre_ansatz[i]]}") 
            print("|0>")  
            if gtol is not None and gnorm < gtol:
                Done = True
                print(f"\nADAPT finished.  (Gradient norm acceptable.)")
                continue
            if max_depth is not None and len(bre_ansatz)+1 > max_depth:
                Done = True
                print(f"\nADAPT finished.  (Max depth reached.)")
                continue
            if Etol is not None and error < Etol:
                Done = True
                print(f"\nADAPT finished.  (Error acceptable.)")
                continue
            iteration += n
            print(f"\nADAPT Iteration {iteration}")


            bre_params2 = []
            block = int(len(bre_params)/n)
            for i in range(0, n):
                bre_params2.append(0)
                bre_params2 += list(bre_params[i*block:(i+1)*block])
            
            bre_params = np.array(bre_params2)
              
            ansatz = [idx[-1]] + ansatz
            bre_ansatz = copy.copy(ansatz)
            for i in range(1, n):
                bre_ansatz += ansatz
            print(f"Recycled ansatz:")
            for i in range(0, len(bre_ansatz)):
                print(f"{i} {bre_params[i]} {self.v_pool[bre_ansatz[i]]}") 
            print("|0>")  

            H_vqe = copy.copy(self.H_vqe)
            pool = copy.copy(self.pool)
            ref = copy.copy(self.ref)
            bre_params = multi_vqe(bre_params, bre_ansatz, H_vqe, pool, ref, self, guesses = guesses, hf = hf, threads = threads)  
            state = t_ucc_state(bre_params, bre_ansatz, self.pool, self.ref)
            np.save(f"{self.system}/bre_params", bre_params)
            np.save(f"{self.system}/bre_ops", bre_ansatz)
        self.e_dict = {}
        self.grad_dict = {}
        self.state_dict = {}
            
        print(f"\nConverged ADAPT energy:    {E:20.16f}")            
        print(f"\nConverged ADAPT error:     {error:20.16f}")            
        print(f"\nConverged ADAPT gnorm:     {gnorm:20.16f}")            
        print(f"\nConverged ADAPT fidelity:  {fid:20.16f}")            
        print("\n---------------------------\n")
        print("\"Adapt.\" - Bear Grylls\n")
        print("\"ADAPT.\" - Harper \"Grimsley Bear\" Grimsley\n")
        repo = git.Repo(search_parent_directories=True)
        sha = repo.head.object.hexsha
        print(f"Git revision:\ngithub.com/hrgrimsl/fixed_adapt/commit/{sha}")
        return error

    def adapt(self, params, ansatz, ref, gtol = None, Etol = None, max_depth = None, criteria = 'grad', guesses = 0, square = False):
        """Vanilla ADAPT algorithm for arbitrary reference.  No sampling, no tricks, no silliness.  
        :param params: Parameters associated with ansatz.
        :type params: list 
        :param ansatz: List of operator indices, applied to reference in reversed order.
        :type ansatz: list
        :param ref: Reference state.
        :type ref: (2^N,1) array-like
        :param gtol: Stopping condition on gradient norm of all ops to add
        :type gtol: float
        :param Etol: Stopping condition on error from ED of all ops
        :type Etol: float
        :param max_depth: Stopping condition on operators
        :type max_depth: int
        """
        self.e_dict = {}
        self.grad_dict = {}
        self.state_dict = {}
        self.e_savings = 0
        self.grad_savings = 0
        self.state_savings = 0
        state = t_ucc_state(params, ansatz, self.pool, self.ref)
        iteration = len(ansatz)
        print(f"\nADAPT Iteration {iteration}")
        print("Performing ADAPT:")
        E = (state.T@(self.H@state))[0,0] 
        Done = False
        while Done == False:

            gradient = 2*np.array([((state.T@(self.H_adapt@(op@state)))[0,0]) for op in self.pool]).real
            gnorm = np.linalg.norm(gradient)
            if criteria == 'grad':
                idx = np.argsort(abs(gradient))
                 
            E = (state.T@(self.H@state))[0,0].real 
            error = E - self.ed_energies[0]
            fid = ((self.ed_wfns[:,0].T)@state)[0,0].real**2
            print(f"\nBest Initialization Information:")
            print(f"Operator/ Expectation Value/ Error")

            for key in self.sym_ops.keys():
                val = ((state.T)@(self.sym_ops[key]@state))[0,0].real
                err = val - self.ed_syms[0][key]
                print(f"{key:<6}:      {val:20.16f}      {err:20.16f}")
                
            print(f"Next operator to be added: {self.v_pool[idx[-1]]}")
            print(f"Operator multiplicity {1+ansatz.count(idx[-1])}.")                
            print(f"Associated gradient:       {gradient[idx[-1]]:20.16f}")
            print(f"Gradient norm:             {gnorm:20.16f}")
            print(f"Fidelity to ED:            {fid:20.16f}")
            print(f"Current ansatz:")
            for i in range(0, len(ansatz)):
                print(f"{i} {params[i]} {self.v_pool[ansatz[i]]}") 
            print("|0>")  
            if gtol is not None and gnorm < gtol:
                Done = True
                print(f"\nADAPT finished.  (Gradient norm acceptable.)")
                continue
            if max_depth is not None and iteration+1 > max_depth:
                Done = True
                print(f"\nADAPT finished.  (Max depth reached.)")
                continue
            if Etol is not None and error < Etol:
                Done = True
                print(f"\nADAPT finished.  (Error acceptable.)")
                continue
            iteration += 1
            print(f"\nADAPT Iteration {iteration}")

            params = np.array([0] + list(params))
            ansatz = [idx[-1]] + ansatz
            H_vqe = copy.copy(self.H_vqe)
            pool = copy.copy(self.pool)
            ref = copy.copy(self.ref)
            params = multi_vqe(params, ansatz, H_vqe, pool, ref, self, guesses = guesses)  
            if square == True:
                multi_vqe_square(params, ansatz, H_vqe, pool, ref, self, guesses = guesses)
            state = t_ucc_state(params, ansatz, self.pool, self.ref)
            np.save(f"{self.system}/params", params)
            np.save(f"{self.system}/ops", ansatz)
        self.e_dict = {}
        self.grad_dict = {}
        self.state_dict = {}
            
        print(f"\nConverged ADAPT energy:    {E:20.16f}")            
        print(f"\nConverged ADAPT error:     {error:20.16f}")            
        print(f"\nConverged ADAPT gnorm:     {gnorm:20.16f}")            
        print(f"\nConverged ADAPT fidelity:  {fid:20.16f}")            
        print("\n---------------------------\n")
        print("\"Adapt.\" - Bear Grylls\n")
        print("\"ADAPT.\" - Harper \"Grimsley Bear\" Grimsley\n")
        repo = git.Repo(search_parent_directories=True)
        sha = repo.head.object.hexsha
        print(f"Git revision:\ngithub.com/hrgrimsl/fixed_adapt/commit/{sha}")


    def graph(self, ansatz, ref):
        import networkx as nx
        import matplotlib.pyplot as plt
        import colorsys
        print("\nAnsatz Structure:\n")
        for i in ansatz:
            print(self.v_pool[i])
        print("|0>")
        print("Identifying all accessible determinants.")
        print(f"Controllable Parameters:  |Determinants Accessible:")
        dets = [copy.copy(ref)]
        j = 0
        print(f"{j:10d}                    {len(dets):10d}")
        for i in reversed(ansatz):
            j += 1
            op = self.pool[i]
            op*= 1/np.linalg.norm((op@ref).todense())
            new_dets = []
            for det in dets:
                new_det = op@det
                if (new_det.T@new_det)[0,0] > .01 and self.is_in(new_det, dets) == False and self.is_in(new_det, new_dets) == False:
                    new_dets.append(copy.copy(new_det))
            op = op@op
            op*= 1/np.linalg.norm((op@ref).todense())
            for det in dets:
                new_det = op@det
                if (new_det.T@new_det)[0,0] > .01 and self.is_in(new_det, dets) == False and self.is_in(new_det, new_dets) == False:
                    new_dets.append(copy.copy(new_det))
            dets += new_dets

            print(f"{j:10d}                    {len(dets):10d}")
        a_mat = np.zeros((len(dets), len(dets)))
        cur_dets = [0]  
        o_mats = []      
        for k in reversed(ansatz):
            o_mat = np.zeros((len(dets), len(dets)))
            op = self.pool[k]
            op*= 1/np.linalg.norm((op@ref).todense())
            new_dets = []
            for i in cur_dets:
                for j in range(0, len(dets)):
                    if abs((dets[i].T@op@dets[j]).todense()[0,0]) > .01:
                            print(dets[i])
                            print(dets[j])
                            print('---')
                            new_dets.append(j)
                            a_mat[i,j] += 1
                            a_mat[j,i] += 1
                            o_mat[i,j] += 1
                            o_mat[j,i] += 1
            o_mats.append(copy.copy(o_mat))
            cur_dets += new_dets
        H_dets = np.zeros((len(dets), len(dets)))

        colors = [colorsys.hsv_to_rgb(float(i+1)/len(ansatz),1.0,1.0) for i in range(0, len(ansatz))]
        print(colors)
        G = nx.Graph()
        for i in range(0, len(dets)):
            for j in range(i, len(dets)):
                H_dets[i,j] = H_dets[j,i] = (dets[i].T@(self.H_vqe)@dets[j]).todense()[0,0]
        print(f"Accessible Space CI: {np.linalg.eigh(H_dets)[0][0]}")
        for i in range(0, len(dets)):
            G.add_node(i)
        for k in range(0, len(o_mats)):
            o_mat = copy.copy(o_mats[k])
            for i in range(0, len(dets)):
                for j in range(i+1, len(dets)):
                    if o_mat[i,j] > 0:
                        G.add_edge(i, j, color = colors[k]) 

        edge_colors = [G[u][v]['color'] for u,v in G.edges()]
        nx.draw(G, edge_color = edge_colors, with_labels = True)
        plt.show()
                        
        '''
        G = nx.from_numpy_matrix(a_mat)
        nx.draw(G, with_labels = True)
        plt.show()
        '''

    def graph_nick(self, ansatz, ref):
        import networkx as nx
        import matplotlib.pyplot as plt
        import colorsys
        fci = self.ed_wfns[:,0]
        fci_dets = []
        fci_coeffs = []
        for i in range(0, len(list(fci))):
            if abs(fci[i])>1e-10:
               fci_dets.append(i)
               fci_coeffs.append(abs(fci[i]))
        idx = np.argsort(fci_coeffs)[::-1]
        fci_dets = list(np.array(fci_dets)[idx])
        fci_coeffs = list(np.array(fci_coeffs)[idx])
        color_map = [plt.cm.gray(1-i/fci_coeffs[0]) for i in fci_coeffs]
        G = nx.Graph()
        pos = nx.circular_layout(G)
        N = len(fci_dets)
        theta = 2*math.pi/N
        for i in range(0, N):
            G.add_node(fci_dets[i],pos=(np.cos(theta*i),np.sin(theta*i)))
        pos=nx.get_node_attributes(G,'pos')
        cur_dets = [fci_dets[0]]
        #print("Assuming ref is most important det.")
        Adj = np.zeros(self.H_vqe.shape)
        for i in reversed(range(0, len(ansatz))):
            Done = False
            while Done == False:
                op = self.pool[ansatz[i]].todense()
                new_dets = []
                for j in cur_dets:
                    for k in fci_dets:
                        if abs(op[j,k]) > .01:
                           Adj[j,k] += .1
                           if k not in new_dets and k not in cur_dets:
                                new_dets.append(k)
                cur_dets += new_dets
                if len(new_dets) == 0:
                    Done = True
        H = np.zeros((len(cur_dets),len(cur_dets)))
        for i in range(0, len(cur_dets)):
            for j in range(i, len(cur_dets)):
                H[i,j] = H[j,i] = self.H_vqe[cur_dets[i],cur_dets[j]]
        print(f"{len(cur_dets)}-determinant subspace ED Energy: {np.linalg.eigh(H)[0][0]}") 
        for i in range(0, self.H_vqe.shape[0]):
            for j in range(0, self.H_vqe.shape[0]):
                if i in fci_dets and j in fci_dets and i != j:
                    G.add_edge(i, j, weight = (0,0,0,Adj[i,j])) 

        weights = nx.get_edge_attributes(G,'weight').values()
        nx.draw(G, pos, edge_color = list(weights), width = [4 for i in list(weights)], node_color = color_map, with_labels = False, verticalalignment = 'top')

        for i in range(0, N):
            plt.text(1.2*np.cos(theta*i),1.2*np.sin(theta*i),s = str(fci_dets[i]))
        ax = plt.gca()
        ax.collections[0].set_edgecolor('black')
        plt.xlim(-1.5,1.5)
        plt.ylim(-1.5,1.5)
        plt.axis('equal')
        plt.savefig(f'img-{len(ansatz)}.pdf', bbox_inches = "tight")
        plt.show()

    def is_in(self, det, dets):
        for det2 in dets:
            if abs((det.T@det2)[0,0]) > .9:
                return True
        return False
    
    


def t_ucc_state(params, ansatz, pool, ref):
    state = copy.copy(ref)
    for i in reversed(range(0, len(ansatz))):
        state = scipy.sparse.linalg.expm_multiply(params[i]*pool[ansatz[i]], state)
    return state

def t_ucc_E(params, ansatz, H_vqe, pool, ref):
    state = t_ucc_state(params, ansatz, pool, ref)
    E = (state.T@(H_vqe)@state).todense()[0,0].real
    return E       

def kup_E(params, k, H_vqe, pool, ref):
    state = copy.copy(ref)
    for j in reversed(range(0, k)):
        G = 0*H_vqe
        for i in range(0, len(pool)):
            param_idx = j*len(pool) + i
            G += params[param_idx]*pool[i]
        state = scipy.sparse.linalg.expm_multiply(G, state)
    E = (state.T@(H_vqe)@state).todense()[0,0].real
    return E
       
def t_ucc_grad(params, ansatz, H_vqe, pool, ref):
    state = t_ucc_state(params, ansatz, pool, ref)
    hstate = H_vqe@state
    grad = [2*((hstate.T)@pool[ansatz[0]]@state).todense()[0,0]]
    hstack = scipy.sparse.hstack([hstate,state]) 
    for i in range(0, len(params)-1):
        hstack = scipy.sparse.linalg.expm_multiply(-params[i]*pool[ansatz[i]], hstack).tocsr()
        grad.append(2*((hstack[:,0].T)@pool[ansatz[i+1]]@hstack[:,1]).todense()[0,0])
    grad = np.array(grad)
    return grad.real

def t_ucc_hess(params, ansatz, H_vqe, pool, ref):
    J = copy.copy(ref)
    for i in reversed(range(0, len(params))):
        J = scipy.sparse.hstack([pool[ansatz[i]]@J[:,-1], J]).tocsr()
        J = scipy.sparse.linalg.expm_multiply(pool[ansatz[i]]*params[i], J)
    J = J.tocsr()[:,:-1]
    u, s, vh = np.linalg.svd(J.todense())       
    hess = 2*J.T@(H_vqe@J).todense()       
    state = t_ucc_state(params, ansatz, pool, ref)
    hstate = H_vqe@state
    for i in range(0, len(params)):            
        hstack = scipy.sparse.hstack([copy.copy(hstate), copy.copy(J[:,i])]).tocsc() 
        for j in range(0, i+1):
            hstack = scipy.sparse.linalg.expm_multiply(-params[j]*pool[ansatz[j]], hstack)
            ij = 2*((hstack[:,0].T)@pool[ansatz[j]]@hstack[:,1]).todense()[0,0]
            if i == j:
                hess[i,i] += ij
            else:
                hess[i,j] += ij
                hess[j,i] += ij
    w, v = np.linalg.eigh(hess)
    energy = ((hstate.T)@state).todense()[0,0]
    grad = ((J.T)@hstate).todense()
    #print(f"Energy: {energy:20.16f}")
    #print(f"GNorm:  {np.linalg.norm(grad):20.16f}")
    #print(f"Jacobian Singular Values:")
    #spec_string = ""
    #for sv in s:
    #    spec_string += f"{sv},"
    #print(spec_string)
    #print(f"Hessian Eigenvalues:")
    #spec_string = ""
    #for sv in w:
    #    spec_string += f"{sv},"
    #print(spec_string)
    return hess.real

def t_ucc_jac(params, ansatz, H_vqe, pool, ref):
    J = copy.copy(ref)
    for i in reversed(range(0, len(params))):
        J = scipy.sparse.hstack([pool[ansatz[i]]@J[:,-1], J]).tocsr()
        J = scipy.sparse.linalg.expm_multiply(pool[ansatz[i]]*params[i], J)
    J = J.tocsr()[:,:-1]
    return J.real


def adapt_vqe(ansatz, H_vqe, pool, ref):
    params = []
    ops = []
    print("Params Energy")
    for i in reversed(range(0, len(ansatz))):
        params = [0] + params
        ops = [ansatz[i]] + ops
        res = vqe(np.array(params), ops, H_vqe, pool, ref)
        params = list(res.x)
        print(f"{len(params)} {res.fun}")
 
def wfn_grid(op, pool, ref, xiphos):
    for i in range(0, 1001):
        x = 2*math.pi*i/1000
        wfn = scipy.sparse.linalg.expm_multiply(x*op, ref)
        c0 = (ref.T@wfn).todense()[0,0]
        print(f"{x} {c0}")
    exit()

def multi_vqe(params, ansatz, H_vqe, pool, ref, xiphos, energy = None, guesses = 0, hf = True, threads = 1):
    os.system('export OPENBLAS_NUM_THREADS=1')
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    if energy is None or energy == t_ucc_E:
        energy = t_ucc_E
        jac = t_ucc_grad
        hess = t_ucc_hess
    param_list = [copy.copy(params)]
    seeds = ['Recycled']
    if hf == True:
        seeds.append('HF')
        param_list.append(0*params)
    for i in range(0, guesses):
        seed = i+guesses*(len(params)-1)
        seeds.append(seed)
        np.random.seed(seed)
        param_list.append(math.pi*2*np.random.rand(len(params)))
        #E0s.append(energy(param_list[-1], ansatz, H_vqe, pool, ref))

    #iterable = [*zip(param_list, [ansatz for i in range(0, len(param_list))], [H_vqe for i in range(0, len(param_list))], [pool for i in range(0, len(param_list))], [ref for i in range(0, len(param_list))], E0s, [xiphos for i in range(0, len(param_list))]] 
    iterable = [*zip(param_list, [ansatz for i in range(0, len(param_list))], seeds, [xiphos for j in range(0, len(param_list))])]

    start = time.time()
    with Pool(threads) as p:
        L = p.starmap(detailed_vqe, iterable = iterable)
    print(f"Time elapsed over whole set of optimizations: {time.time() - start}")
    #params = solution_analysis(L, ansatz, H_vqe, pool, ref, seeds, param_list, E0s, xiphos)
    params = L[0][0].x
    idx = np.argsort([L[i][0].fun for i in range(0, len(L))])
    for i in idx:
        print(L[i][1], flush = True)
    return params

def full_scan(ansatz, H_vqe, pool, ref, xiphos, gridpoints = 100):
    from multiprocessing import Pool
    params = np.zeros(len(ansatz))
    grid_pts = [params]
    one_d_grids = []
    for i in range(0, len(ansatz)):
       one_d_grids.append()
    print(one_d_grids)   

def multi_kup(H_vqe, ref, xiphos, k = 1, guesses = 0):
    #does uccsd-type by default
    from multiprocessing import Pool
    start = time.time()
    os.system('export OPENBLAS_NUM_THREADS=1')
    os.environ['OPENBLAS_NUM_THREADS'] = '1'

    param_list = [k*np.zeros(len(xiphos.pool))]
    seeds = ['HF']
    for i in range(0, guesses):
        seed = i+guesses*(k*len(xiphos.pool)-1)
        seeds.append(seed)
        np.random.seed(seed)
        param_list.append(math.pi*2*np.random.rand(k*len(xiphos.pool)))
    iterable = [*zip(param_list, seeds, [k for j in range(0, len(param_list))], [xiphos for j in range(0, len(param_list))])]
    with Pool(126) as p:
        L = p.starmap(detailed_kup, iterable = iterable)
    print(f"Time elapsed over whole set of optimizations: {time.time() - start}")
    #params = solution_analysis(L, ansatz, H_vqe, pool, ref, seeds, param_list, E0s, xiphos)
    params = L[0][0].x
    idx = np.argsort([L[i][0].fun for i in range(0, len(L))])
    for i in idx:
        print(L[i][1], flush = True)
    return params

def multi_vqe_square(params, ansatz, H_vqe, pool, ref, xiphos, energy = None, guesses = 0):
    from multiprocessing import Pool
    start = time.time()
    os.system('export OPENBLAS_NUM_THREADS=1')
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    if energy is None or energy == t_ucc_E:
        energy = t_ucc_E
        jac = t_ucc_grad
        hess = t_ucc_hess
    param_list = [np.array(list(0*params)+list(copy.copy(params)))]
    seeds = ['Recycled']

    for i in range(0, guesses):
        seed = 10000000 + i + guesses*(len(params)-1)
        seeds.append(seed)
        np.random.seed(seed)
        param_list.append(math.pi*2*np.random.rand(2*len(params)))

    iterable = [*zip(param_list, [ansatz + ansatz for j in range(0, len(param_list))], seeds, [xiphos for j in range(0, len(param_list))])]
    with Pool(126) as p:
        L = p.starmap(detailed_vqe_square, iterable = iterable)
    print(f"Time elapsed over whole set of optimizations: {time.time() - start}")
    #params = solution_analysis(L, ansatz, H_vqe, pool, ref, seeds, param_list, E0s, xiphos)
    params = L[0][0].x
    idx = np.argsort([L[i][0].fun for i in range(0, len(L))])
    for i in idx:
        print(L[i][1], flush = True)
    return params



def detailed_kup(params, seed, k, xiphos):
    energy = kup_E
    x0 = params
    E0 = energy(params, k, xiphos.H_vqe, xiphos.pool, xiphos.ref)
    res = scipy.optimize.minimize(energy, params, method = "bfgs", args = (k, xiphos.H_vqe, xiphos.pool, xiphos.ref), options = {'gtol': 1e-8})
    EF = res.fun
    string = "\nSolution Analysis:\n\n"
    string += f"Total Parameters: {len(params)}\n"
    string += f"Initialization: {seed}\n"
    string += f"Initial Energy: {E0:20.16f}\n"
    string += f"Final Energy: {EF:20.16f}\n"
    string += '\n\n'
    return [res, string]

def vqe(params, ansatz, H_vqe, pool, ref, strategy = "bfgs", energy = None):
    if energy is None or energy == t_ucc_E:
        energy = t_ucc_E
        jac = t_ucc_grad
        hess = t_ucc_hess
    if strategy == "newton-cg":
        res = scipy.optimize.minimize(energy, params, jac = jac, hess = hess, method = "newton-cg", args = (ansatz, H_vqe, pool, ref), options = {'xtol': 1e-16})
    if strategy == "bfgs":
        res = scipy.optimize.minimize(energy, params, jac = jac, method = "bfgs", args = (ansatz, H_vqe, pool, ref), options = {'gtol': 1e-8})
    return res


    

def detailed_vqe(params, ansatz, seed, xiphos, jac_svd = False, hess_diag = False):
    energy = t_ucc_E
    jac = t_ucc_grad
    hess = t_ucc_hess
    x0 = params
    E0 = energy(params, ansatz, xiphos.H_vqe, xiphos.pool, xiphos.ref)
    res = scipy.optimize.minimize(energy, params, jac = jac, method = "bfgs", args = (ansatz, xiphos.H_vqe, xiphos.pool, xiphos.ref), options = {'gtol': 1e-8})
    #res = scipy.optimize.minimize(energy, params, jac = jac, method = "bfgs", args = (ansatz, xiphos.H_vqe, xiphos.pool, xiphos.ref), options = {'gtol': 1e-16})
    EF = res.fun
    #gradient = t_ucc_grad(res.x, ansatz, xiphos.H_vqe, xiphos.pool, xiphos.ref)
    gradient = res.jac
    gnorm = np.linalg.norm(gradient)
    state = t_ucc_state(res.x, ansatz, xiphos.pool, xiphos.ref)
    fid = ((xiphos.ed_wfns[:,0].T)@state)[0,0].real**2
    string = "\nSolution Analysis:\n\n"
    string += f"Parameters: {len(ansatz)}\n"
    string += f"Initialization: {seed}\n"
    string += f"Initial Energy: {E0:20.16f}\n"
    string += f"Final Energy: {EF:20.16f}\n"
    string += f"GNorm: {gnorm:20.16f}\n"
    string += f"Fidelity: {fid:20.16f}\n"
    string += f"Success: {res.success}\n"
    string += f"Energy Evals: {res.nfev+1}\n"
    string += f"Gradient Evals: {res.njev}\n"
    string += f"Initial Parameters:\n"
    for x in x0:
        string += f"{x},"
    string += "\n"
    string += f"Solution Parameters:\n"
    for x in res.x:
        string += f"{x},"
    string += "\n"
    if jac_svd == True:
        jacobian = t_ucc_jac(res.x, ansatz, xiphos.H_vqe, xiphos.pool, xiphos.ref)
        u, s, vh = np.linalg.svd(jacobian.todense())
        string += f"Jacobian Singular Values:\n"
        for sv in s:
            string += f"{sv},"
        string += "\n"
    if hess_diag == True:
        hessian = t_ucc_hess(res.x, ansatz, xiphos.H_vqe, xiphos.pool, xiphos.ref)
        w, v = np.linalg.eigh(hessian)
        string += f"Hessian Eigenvalues:\n"
        for sv in w:
            string += f"{sv},"
        string += "\n"
    string += f"Operator/ Expectation Value/ Error:\n"
    for key in xiphos.sym_ops.keys():
        val = ((state.T)@(xiphos.sym_ops[key]@state))[0,0].real
        err = val - xiphos.ed_syms[0][key]
        string += f"{key:<6}:      {val:20.16f}      {err:20.16f}\n"
    string += '\n\n'
    return [res, string]

def solution_analysis(L, ansatz, H_vqe, pool, ref, seeds, param_list, E0s, xiphos, guess = 'recycled'):
    Es = [L[i].fun for i in range(0, len(L))]
    xs = [L[i].x for i in range(0, len(L))]
    gs = [t_ucc_grad(L[i].x, ansatz, H_vqe, pool, ref) for i in range(0, len(L))]
    hess = [t_ucc_hess(L[i].x, ansatz, H_vqe, pool, ref) for i in range(0, len(L))]
    jacs = [t_ucc_jac(L[i].x, ansatz, H_vqe, pool, ref) for i in range(0, len(L))]
    idx = np.argsort(Es)
    print(f"\nSolution Analysis:\n")
    smins = []
    for i in idx:
        seed = seeds[i]
        E0 = E0s[i]
        EF = Es[i]
        g_norm = np.linalg.norm(gs[i])
        u, s, vh = np.linalg.svd(jacs[i].todense())       
        s_min = np.min(s)
        smins.append(np.min(s))
        w, v = np.linalg.eigh(hess[i])
        e_min = 1/np.min(w)
        state = t_ucc_state(xs[i], ansatz, pool, ref)
        fid = ((xiphos.ed_wfns[:,0].T)@state)[0,0].real**2
        print(f"Parameters: {len(ansatz)}")
        print(f"Initialization: {seed}")
        print(f"Initial Energy: {E0:20.16f}")
        print(f"Final Energy:   {EF:20.16f}")
        print(f"GNorm:          {g_norm:20.16f}")
        print(f"Fidelity:       {fid:20.16f}")
        print(f"Solution Parameters:")
        spec_string = ""
        for x in xs[i]:
            spec_string += f"{x},"
        print(spec_string)
        print(f"Jacobian Singular Values:")
        spec_string = ""
        for sv in s:
            spec_string += f"{sv},"
        print(spec_string)
        print(f"Hessian Eigenvalues:")
        spec_string = ""
        for sv in w:
            spec_string += f"{sv},"
        print(spec_string)
        print(f"Operator/ Expectation Value/ Error")
        for key in xiphos.sym_ops.keys():
            val = ((state.T)@(xiphos.sym_ops[key]@state))[0,0].real
            err = val - xiphos.ed_syms[0][key]
            print(f"{key:<6}:      {val:20.16f}      {err:20.16f}")
        print('\n')
    if guess == 'recycled':
        return xs[0]
    elif guess == 'best':
        return xs[idx[0]]


