#include("../src/Dbcv.jl")
import Pkg

Pkg.add("Graphs")
Pkg.add("SimpleWeightedGraphs")
Pkg.add("NearestNeighbors")
Pkg.add("Combinatorics")

push!(LOAD_PATH, "../src")
import Dbcv, DelimitedFiles

dataset_file::String, clustering_file::String = ARGS

dataset::AbstractArray = DelimitedFiles.readdlm(dataset_file, ',', Float64)

clustering::AbstractArray = DelimitedFiles.readdlm(clustering_file, ',', Int)

result::Real = Dbcv.dbcv(dataset, vec(clustering[:, 1]))

print(result)

return result


