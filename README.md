# G4S: Free HPC experts from scientific computing programming with graph-based engine
&#160; &#160; &#160; &#160; G4S is a graph-based engine paradigm for modern scientific computing. G4S uses a unified graph programming interface to implement matrix computation, without having to deal with the complexity of programming on a large-scale computing platform. The G4S paradigm introduces a tool to view matrix operations from the graph perspective. Different matrix operation cases can be efficiently executed on large-scale computing platforms using the existing graph engine to automatically explore the optimal execution strategies. We have implemented three typical scientific computing routines using the G4S paradigm: DeePMD-kit, Citcoms, and Cantera.

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

#### citcoms

```
    [G4S] cd $citcoms_dir
    [G4S] ./configure
    [G4S] cd $citcoms_dir/examples/Cookbook2
    [G4S] CitcomSRegional cookbook2
```

## Data Description

### deepMD-kit
    deepMD-kit is a toolkit for molecular dynamics simulation and machine learning. It includes pre-trained models and example datasets.
#### water
The Water dataset is a collection of data used to describe the water molecule system. It includes a series of atomic coordinates and corresponding information such as energy and force fields. This dataset can be utilized for training and testing molecular models based on deep learning methods. It provides a representative structure of a small molecule system, making it suitable for fast algorithm verification and model validation.
#### cuprum
The Cuprum dataset is a collection of data used to describe the copper metal system. It includes a series of atomic coordinates, lattice constants, and corresponding energy information. This dataset can be used for training and testing molecular models based on deep learning methods. It provides a representative structure of a metal system, making it suitable for studying the properties and behavior of copper metal.
#### Fe-H 
The Fe-H dataset is a collection of data used to describe the iron-hydrogen system. It includes a series of atomic coordinates, lattice constants, and corresponding energy information. This dataset can be utilized for training and testing molecular models based on deep learning methods. It provides a representative structure of metal-hydrogen atomic interactions, making it suitable for studying the properties and interactions of iron-hydrogen systems.

## citcoms
CitcomS is a finite element code designed to solve compressible thermochemical convection problems relevant to Earth's mantle.
### CookBook 1
This example solves for thermal convection within a full spherical shell domain.  The full spherical version of CitcomS.py is designed to run on a cluster that decomposes the spherical shell into 12 equal “caps” and then distributes the calculation for caps onto separate processors.  To run CitcomS.py with the full solver parameter set, it is recommended that you have a minimum of 12 processors available on your cluster.
### CookBook 2
This example solves for thermal convection with velocity boundary conditions imposed on the top surface within a given region of a sphere. This requires using the regional version of CitcomS.py.
This model allows you to create a plate-driven convection in which there is a thermal upwelling on one wall, a thermal downwelling on another, and uniform horizontal velocity across the top. The downwelling is not exactly subduction because the default boundary conditions are close to zero shear stress on the boundaries. This means that there is a symmetrical downwelling in a vertical domain on the other side.

### CookBook 3

A common problem in geophysics is the exploration of natural convection
in the presence of variable viscosity, including temperature-dependent
or stress-dependent viscosity.

## cantera

### NonIdealShockTube
The NonIdealShockTube.py dataset serves as a valuable resource for simulating non-ideal shock tube experiments. Utilizing the power of the Cantera software library, this dataset employs numerical simulations to investigate the intricate gas dynamics within such experiments. By delving into critical phenomena like shock propagation, reactions, and energy conversion, researchers gain a deeper understanding of non-ideal shock tube behavior.