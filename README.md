# G4S: Free HPC experts from scientific computing programming with graph-based engine
&#160; &#160; &#160; &#160; 

## Dependencies, Compilation and Running
### 1. External Dependencies
#### Software Dependencies
Before running G4S codes, it's essential that you have already install software dependencies shown below.
```
    g++ (GCC) 4.8.5
    python ($>=$3.7)
    pip=21.1
    scons
    boost-cpp
    cython
    numpy
    pytest
    matplotlib
    ipython
    mpich
    fort77
    mpi4py
```

#### Data Dependencies
We evaluate G4S using three specific benchmarks, deepMD-kit, citcoms, and cantera. The source code for the benchmark is included in our codebases (in the *G4S/cantera*, *G4S/citcoms* and *G4S/deepmd* folders).

### 2. Running, and Performance Testing
G4S can be tested for performance using deepmd, citcoms and cantera. Since G4S does not support checkpointing, we had to re-run each benchmark to populate the database for each test.

#### deepMD-kit
    
```
    [G4S] conda activate /path/to/deepmd-kit
    [G4S] cd $deepmd_source_dir/examples/water/se_e2_a/
    [G4S] dp train input.json
    [G4S] dp freeze -o graph.pb
    [G4S] dp compress -i graph.pb -o compress.pb
    [G4S] cd $deepmd_source_dir/examples/water/lmp
    [G4S] lmp -in in.lammps 
```

#### cantera

    
```
    [G4S] conda activate /path/to/ct-build
    [G4S] cd $cantera_source_dir/interfaces/cython/cantera/examples/reactors
    [G4S] python  NonIdealShockTube.py
```