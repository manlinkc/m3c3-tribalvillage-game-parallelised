# M3C Project 3: Tribal Village Game Parallelised
Project 3: ‘Tribal Competition Game Parallelization’ completed for ‘High Performance Computing’ module.

The code simulates a game consisting of an N x N grid where each cell represents a village. Within each iteration a village changes to either a ‘Mercenaries’ or ‘Collaborators’ based on randomly generated probabilities. The affiliation of a cell influences the affiliation of neighbouring cell. This code was used the simulate the game over multiple iterations. Three implementations of this game were written, one is a Python implementation, a Fortran implementation and a Fortran+OpenMP implementation which involves parallelisation to optimise the performance.  The function `performance` in the file `hw3_dev.py` was used to generate plots (included in the repo) that were used assess/compare the performance of each implementation. A detailed analysis is included in the docstring. 

