adapt
==============================
[//]: # (Badges)
[![GitHub Actions Build Status](https://github.com/hrgrimsl/adapt/workflows/CI/badge.svg)](https://github.com/REPLACE_WITH_OWNER_ACCOUNT/adapt/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/hrgrimsl/adapt/branch/master/graph/badge.svg)](https://codecov.io/gh/REPLACE_WITH_OWNER_ACCOUNT/adapt/branch/master)


A Python package to run ADAPT-VQE.

Hello, and welcome to my ADAPT code!

Check out template.py to see how to run your own calculations!

The main files of interest to users:

The pyscf_backend.py file contains tools to handle converting basic chemical information into useful integrals, etc.

The of_translator.py file contains tools to convert molecular integrals into operators in the Hilbert space basis used in driver.py.

The system_methods.py file mainly contains tools to get operator pools of anti-Hermitian generators.

The computational_tools.py file contains miscellaneous computational tools, mostly for random stuff I wanted to try.

The driver.py file contains the actual algorithms used in our work. The breadapt() function is probably the main thing of interest. ("BREADAPT" is what I originally called ADAPT^N.) Using n = 1, the default, will give traditional ADAPT results. The value of n can be increased to help with local minima if desired, as we discuss in our paper. The t_ucc_energy(), t_ucc_grad(), and t_ucc_hess() functions give efficient ways to compute an ADAPT-VQE energy, gradient, and Hessian at a given step. The detailed_vqe() function is used to do each actual VQE. The t_ucc_state() function is useful for computing a statevector given an operator sequence.

Please feel free to reach out for more details at:

hrgrimsl@vt.edu





### Copyright

Copyright (c) 2022, Harper Grimsley


#### Acknowledgements
 
Project based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.6.
