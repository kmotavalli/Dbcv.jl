push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
import Dbcv, DelimitedFiles

has_header::Integer = 0
dataset::AbstractArray = []
dataset_file::String, clustering_file::String = ARGS

if length(ARGS) == 3
    has_header = parse(Int, ARGS[3])
end

if has_header > 0
    dataset = DelimitedFiles.readdlm(dataset_file, ',', BigFloat, skipstart=1)
else
    dataset = DelimitedFiles.readdlm(dataset_file, ',', BigFloat)
end

clustering::AbstractArray = DelimitedFiles.readdlm(clustering_file, ',', Int)

result::Real = Dbcv.dbcv(dataset, vec(clustering[:, 1]))

print(result)