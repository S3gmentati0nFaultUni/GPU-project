# Introduction
The repository contains the code associated with a parallel MSTp solver I implemented as well as the code for the latex report written for the project. For the project I decided to follow what was written in the paper <a href="https://ieeexplore.ieee.org/document/7092783">A generic and highly efficient parallel variant of Boruvka's algorithm</a>. The solution suffers from a series of problems that I discovered during the project and for which I proposed a possibile solution in the conclusions section.

# Compilation and execution
The code was developed in the "free for use" development platform from Google, Colab.

The whole notebook can be imported in the environment and run, should already contain everything it needs to work, as far as tests are concerned two different approaches can be followed:

- A graph generator, mind that if the graph is not sparse the performance of the algorithm will fail to meet expectations
- The graphs proposed in the 9th DIMACS challenge (<a href="https://diag.uniroma1.it/challenge9/">Benchmarks</a>) due to the weird format of the data they should be cleaned with something like a python script or similar because the reading script is expecting a file very similar to a csv.

The conclusions for the project can be found in the <a href="https://github.com/S3gmentati0nFaultUni/GPU-project/releases/download/final-release/report.pdf">report</a>
