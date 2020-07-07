## Source Code

This repository contains code used to generate figures for our paper on mechanosensing as an adaptation for neutrophil polarity. Code for the 2D simulations can be made available upon request.

Written by Cole Zmurchok, <a href="https://zmurchok.github.io">https://zmurchok.github.io</a>.

## Requirements

Python, Matlab, XPPAUT, and the matlab interface for plotting xpp diagrams: http://www.math.pitt.edu/~bard/xpp/plotxppaut4p4.m

## Contents

Each file generates a part of a figure:

- Fig 2:
    - bettercolors.m is a Matlab script that changes the default Matlab colors.
    - Fig2LPA.ode is the XPPAUT (http://www.math.pitt.edu/~bard/xpp) file used to produce the LPA bifurcation diagram. The data from this diagram is saved as diagram_data.dat and is imported into the Matlab figure Fig2a.fig for plotting refinements.
    - Fig2b-e.py is a Python file used to solve the reaction-diffusion PDE to produce the remaining panels in Figure 2. Scipy, Matplotlib and Numpy are needed. This code was based off of the tutorial at <a href="https://docs.scipy.org/doc/scipy/reference/tutorial/integrate.html#mol">https://docs.scipy.org/doc/scipy/reference/tutorial/integrate.html#mol</a>
- Fig 3:
    - Fig3b.py generates Figure 3B by solving the RD-PDE with time-dependent parameters.
- Fig 4:
    - Fig4b.py, Fig4c.py, and Fig4d.py by solving the moving-boundary RD-PDE.
- Fig 5:
   - 'X_data.mat' is a Matlab data file containing data from Python simulations of the reaction-diffusion PDE system for different stimulus levels X.
   - Each data file is produced by the Python script pdeSimulation.py
   - Fig5LPA.ode is the XPPAUT file used to generate the LPA bifurcation diagram (data saved in Fig5LPA.dat and in the Matlab figure Fig5LPA.fig)
   - Fig5a.m and Fig5b.m generate panels A and B respectively.You will need to update the path to Fig5LPA.fig in the Fig5a.m file.
   - is_polarized.m is a function used to determine if a spatial distribution of Rac is polarized or not at a single time step.
- Fig7:
   - bGrad.py solves the moving-boundary RD-PDE in a chemoattractant gradient for various feedback strength parameter values.
   - bGrad_plotter.py plots the results.
- SMFig1
   - Fig2LPA.ode is used again (since Fig 2 is built off of this step). Data from XPPAUT is saved as diagram.dat and SILPAFig.fig.
   - FigSILPA.m is used for plotting refinements.
- SMFig2
   - SIFig2.py solves the RD PDE as in Figure 4B with successively increasing numbers of grid points and plots the mass conservation error for each simulation.

## License

GNU GPL v3 (see License.txt)
