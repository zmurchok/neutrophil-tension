## Source Code

This repository contains code used to generate figures for our paper on mechanosensing as an adaptation for neutrophil polarity.

Written by Cole Zmurchok, <a href="https://zmurchok.github.io">https://zmurchok.github.io</a>.

## Contents

Each file generates a part of a figure:

- bettercolors.m is a Matlab script that changes the default Matlab colors.
- Figure2a_driver.m generates the LPA bifurcation diagram in Figure 2A. It requires the file Fig2_Functions.m (that has the ODE system) and the software <a href="https://sourceforge.net/projects/matcont/">Matcont</a>
- Fig2b-e.py is a Python file used to solve the reaction-diffusion PDE to produce the remaining panels in Figure 2. Scipy, Matplotlib and Numpy are needed. This code was based off of the tutorial at <a href="https://docs.scipy.org/doc/scipy/reference/tutorial/integrate.html#mol">https://docs.scipy.org/doc/scipy/reference/tutorial/integrate.html#mol</a>
- Fig3a.m generates Figure 3A, and requires the file Fig2_Functions.m
- Fig3b.py generates Figure 3B by solving the RD-PDE as before.
- Fig4b.py and Fig4c.py by solving the moving-boundary RD-PDE.
- The directory Fig5/ contains several files:
   - 'X_data.mat' is a Matlab data file containing data from Python simulations of the reaction-diffusion PDE system for different stimulus levels X.
   - Each data file is produced by the Python script pdeSimulation.py
   - The File RT_2.m generates the bifurcation diagram shown in Figure 5a using Matcont and the function file funcs.m, and saves it as RT_2.fig.
   - Fig5a.m and Fig5b.m generate panels A and B respectively. Fig4a requires the file RT_2.fig. You will need to update the path to RT_2.fig in the Fig5a.m file.
   - is_polarized.m is a function used to determine if a spatial distribution of Rac is polarized or not at a single time step.
- The directory Fig6/ contains data from the 2D simulations.
  - Fig6data.csv is the simulation data (time, area, mass, max - min Rac activity), and dataFig6b.mat is a Matlab version of this csv file that contains column vectors of each of these quantities.
  - crop.m crops the simulation output images to be a consistent size to make the subpanels of Figure 6A.
  - Fig6b.m generates the plots used in Figure 6B from the data in dataFig6b.mat.
- SIFig1.py solves the RD PDE as in Figure 4B with successively increasing numbers of grid points.

## License

GNU GPL v3 (see License.txt)
