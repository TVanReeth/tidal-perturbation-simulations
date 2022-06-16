# tidal-perturbation-simulations
Routines to simulate and visualize tidal perturbations of g-mode pulsations in a circular synchronized binary system, within the framework of the traditional approximation of rotation (TAR).

Python packages required to run the code, are:
- astropy
- matplotlib
- numpy

Other required software packages are:
- the stellar pulsation code GYRE (https://gyre.readthedocs.io/; we rely on its implementation of the TAR)
    

To simulate tidally perturbed g-mode pulsations, adapt the contents of the inlist 'inlist.dat' as needed, and execute the command:

    $ python tidal_perturbation_in_circular_binary.py


To visualize the results, adapt the paths in the script "plot_tidal_perturbation_model.py", and execute the command:

    $ python plot_tidal_perturbation_model.py
    
    
For questions and feedback, please contact: timothy.vanreeth at kuleuven.be

