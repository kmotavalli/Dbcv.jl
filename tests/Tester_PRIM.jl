push!(LOAD_PATH, "../src")
import Dbcv, DelimitedFiles

has_header::Integer = 0
dataset::AbstractArray = []
dataset_file::String, clustering_file::String = ARGS

if size(ARGS, 1) == 3
    has_headers = ARGS[3]
end

if has_header != nothing && has_header == 1
    dataset = DelimitedFiles.readdlm(dataset_file, ',', BigFloat, skipstart=1)
else
    dataset = DelimitedFiles.readdlm(dataset_file, ',', BigFloat)
end
clustering::AbstractArray = DelimitedFiles.readdlm(clustering_file, ',', Int)

result::Real = Dbcv.dbcv(dataset, vec(clustering[:, 1]))

print(result)