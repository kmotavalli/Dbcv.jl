# Dbcv

<!-- Title -->
<h1 align="center">
Dbcv.jl
</h1>

<!-- description -->
<p align="center">
  <strong>Density-Based Clustering Validation index implementation</strong>
</p>


[![Build Status](https://github.com/kmotavalli/Dbcv.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/kmotavalli/Dbcv.jl/actions/workflows/CI.yml?query=branch%3Amain)

Dbcv.jl is a Julia package implementing the DBCV metric, used to map the clustering of a dataset (as obtained, for exapmle, via the DBSCAN clustering algorithm) to an index between -1, for poorly formed clusters or wrong points-to-cluster assignment, to +1, if presented with an obtimal clusterization of the dataset.

It identifies clusters via density variations in relation to the between-clusters density.

This package for Julia tries to be similar in usage to felsiq-dbcv for Python (https://github.com/FelSiq/DBCV) supporting similar options, with difference better documented below, and more importantly, tries to match the calculated index value with the one derived by felsiq-dbcv, when evaluating the same dataset and classification.

In tests, the maximum divergence between Dbcv.jl and felsiq/dbcv on the same input data is in the order of 1*e^-16, often 0.0

The git branch "with-optional-validation-tests" contains the python and julia testing infrastructure to compare the correctness of results, while the main branch includes just a simple CI/CD test with no dependencies on python. To evaluate the correctness of Dbcv.jl results, checkout the with-optional-validation-tests branch, enter the tests folder, and run ```test_all_datesets.py``` (uses internal julia/scipy Kruskal MST) and/or ```test_all_datasets_prim.py``` (uses Dbcv.jl internal PRIM MST implementation and felsiq/dbcv internal PRIM MST implementation, close to the original Matlab dbcv code by Pablo A. Jaskowiak: https://github.com/pajaskowiak/dbcv).

Python modules sklearn (or scikit-learn), numpy, scipy and mpmath are required to be installed in order to be able to run the validation tests in that branch.
One notable difference is in the definition of a custom threshold distance (defaulting to e^-9) below which points are considered duplicates: this threshold is added back to the Validity Validation Score. In Felsiq, this is done adding a fixed quantity (e^-12) irrespective of the eventual user set threshold. In Dbcv.jl this values is calculated based on the user set threshold at runtime, which appairs to be more a more correct  behaviour to the author, creating divergences in results with felsiq/dbcv in the case of a custom set threshold. An optional parameter felsiq_bugforbug can be set to true to reintroduce the broken behaviour in Dbcv.jl for bug for bug compatible results in that case.

The validation suite in python+julia runs on Windows, Mac OS X, GNU/Linux and FreeBSD, and possibly other *nix systems provided the above mentioned python dependecies are available.

Beware that by default felsiq/dbcv uses Kruskal via Scipy, while we default to using the internal PRIM implementation for closeness with the original Dbcv paper implementation. Results between felsiq/dbcv and Dbcv.jl must be compared between the same MST algorithm. Different Minimum Spanning Tree algorithms, sometimes even different implementations ordering results differently, will largely impact the calculated Dbcv index.

## Installation instructions

Once the packages get, hopefully, registred in Julia Packages, you will be able to install it by running the Julia REPL and invoking
```julia
julia> using Pkg

julia> Pkg.add("Dbcv")
```

If not using the Julia package registry or testing this code before its submission request, clone the repository and reference src/Dbcv.jl inside your Julia code with its relative or absolute path, like this in the case the src directory is parent of the one containing your own julia sources:

```julia
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
import Dbcv
```

to simply specify an absolute path irrespective of the current working directory ```@__DIR__```, simply specify the full path without joinpath:

```julia
push!(LOAD_PATH, "/path/to/dbcv/folder/src")
import Dbcv
```


## Usage instructions

import Dbcv in your Julia sources via

```julia
import Dbcv```

then invoke Dbcv.dbcv passing it a dataset and a classification in clusters of that dataset.

```julia
result = Dbcv.dbcv(dataset, classification)```

the two must already be in memory (bound to a variable). The classification vector must only include the classification of the points, in the same order they appear in the dataset, to a cluster id. The dataset must not be repeated in the classification variable. If it is, you can extract with a view from your classification matrix, containing only the column with the cluster ids for the original data. If files have to be read first, eg, from csv files, you have to read them into memory before calling Dbcv.dbcv, for example via the Julia package DelimitedFiles.

Here follows an example that combines csv reading and extracting only the relevant column (the last one) from the classification file, then chosing to use Kruskal instead of the default PRIM MST:

```julia
import Dbcv, DelimitedFiles
has_header::Integer = 0
dataset::AbstractArray = []
dataset_file::String, clustering_file::String = ARGS

if has_header > 0
    dataset = DelimitedFiles.readdlm(dataset_file, ',', BigFloat, skipstart=1)
else
    dataset = DelimitedFiles.readdlm(dataset_file, ',', BigFloat)
end

clustering::AbstractArray = DelimitedFiles.readdlm(clustering_file, ',', Int)

result::Real = Dbcv.dbcv(dataset, vec(clustering[:, 1]), use_libgraphs_kruskal=true)

print(result)
```

Note that the code of Dbcv.jl is multithreaded, particurarly benefiting from classifications to a large number of clusters (each cluster gets evalued parallely on a separate thread), even if this is not the only implemented parallelism. 
To benefit from multithreading, you must start Julia specifiying the number of available threads to the interpreter/VM, like

```
julia --threads 16
```

or set the environment variable ```JULIA_NUM_THREADS``` on your system.


## Supported options

Optional named parameters that can be passed to Dbcv.dbcv() are:

```check_duplicates```, defaults to true: whatever to check the dataset for duplicate points (with pairwise distances below the set sep_threshold). Dbcv exits with error if duplicate points are found.

```noise_id```, integer, the cluster id assigned to "noise" points by the classification algorithm used and which is being evaluated. Defaults to considering the '-1' cluster id as the noise cluster id.

```metric```, defaulting to "SqEuclidean" for squared euclidean. The metric is used to calculate pairwise (each point to each point) distances in a cluster and to filter out duplicate points which are closer to each other than the set threshold. Supported metrics are those defined by the Julia package Distances, see the table at https://github.com/JuliaStats/Distances.jl#distance-type-hierarchy
The "type name" as written in Distances.jl docs must be provided as value of the optional named parameter "metric". Note that not all metrics make sense for all datasets and some may just cause errors if incompatible with the data.

```sep_threshold```, the minimum separation distance between points below which to consider points duplicates if check_duplicates is set to true (its default). This value is also indirectly used to clamp (filter) matrices removing smaller values and gets added back, 3 orders of magnitude smaller, to the final result in the step calculating the VCS (Validity Validation Score). Defaults to 1*e^-9.

```felsiq_bugforbug``` felsiq/dbcv does add back a fixed quantity (1E^-12) in the VCS score, but this does not change when the user sets a custom sep_threshold. We default to deriving that value at runtime based on the user set sep_threshold for correctness, but also implement felsiq/dbcv behaviour (always add 1E^-12) when the optional named parameter felsiq_bugforbug is set to true. defaults to false.

```bits_of_precision```, defaults to 512 for compatibility with felsiq-dbcv. It sets the BigFloat type to used a fixed precision instead of dynamically adjusting the precision at runtime, which is otherwise the Julia behaviour. Chosing a bit size compatible with native cpu floating point lenght and instructions will problably speed up execution, reducing the need to split an arithmetic operation over multiple instructions and cpu clock cycle.

```use_libgraphs_kruskal```, default to false. Whatever to use Krustak as the MST to get the minimum spanning tree of points in a cluster, as provided by the Julia Package Libgraphs, or the interal PRIM MST implementation, close to the original code in Matlab by the Dbcv paper author. Note that while felsiq/dbcv also provides an option in that regarding (internal prim or krustal via scipy), felsiq/dbcv defaults to using Kruskal.