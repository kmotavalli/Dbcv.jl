module Dbcv

    import Graphs, SimpleWeightedGraphs, NearestNeighbors, Combinatorics, Distances
    using Base.Threads

    export dbcv

    function convert_singleton_clusters_to_noise!(y, noise_id)
        cluster_ids = unique(y)
        cluster_sizes = [count(==(id), y) for id in cluster_ids]
        non_noise_ids::AbstractArray{Bool} = similar(y, Bool)
        non_noise_ids .= true
        for i in eachindex(y)
            pos = findfirst(id -> id == y[i], cluster_ids)
            if cluster_sizes[pos] == 1
                #i from the iterator can be a multidimensional coordinates tuple if y is multidim
                y[i] = noise_id
                non_noise_ids[i] = false
            end
        end
        return non_noise_ids
    end

    function dense_ranking(y::AbstractVector)
        vals = sort(unique(y))
        return searchsortedfirst.(Ref(vals), y)
    end

    function check_duplicated_samples(X::AbstractArray; threshold::Real = 1e-9)
        if size(X, 1) <= 1
            return nothing
        end
        knn_tree = NearestNeighbors.KDTree(X, Distances.Euclidean())

        for i in 1:size(X, 1)
            _, dists = NearestNeighbors.nn(knn_tree, X[i,:])
            if any(dist -> dist < threshold, dists)
                throw(FieldError("Duplicate samples have been found in X. Try changing sep_threshold (def e^-9)"))
            end
        end
    end

    function pairwise_self_to_infinity!(X::AbstractArray)
        lastdim = minimum(size(X))
        if lastdim == 0
            return X #nothing
        end

        dims = size(X)
        tot_jump = 1
        cur_jump = 1
        for dim in 1:(ndims(X) - 1)
            cur_jump *= dims[dim]
            tot_jump += cur_jump
        end

        for i in 1:lastdim
            access = 1 + (i - 1) * tot_jump
            X[access] = Inf
        end
    end


    function pair_to_pair_distances(X::AbstractArray;
        metric::AbstractString=squeclidean,
        threshold::Real=1e-9)::AbstractArray

        tolerance=threshold/1e-3
        
        metric_instance = try
        	metric_sym = Symbol(metric)
        	metric_class = getfield(Distances, metric_sym)
		#instanciate
		try
			metric_class(tolerance)
		catch
			print("tolerance not supported by " * metric * "\n")
			metric_class()
		end
	catch e
		error("Metric " * metric * " not found in the Distances.jl package, is it spelled correctly?\n
			see https://github.com/JuliaStats/Distances.jl?tab=readme-ov-file#distance-type-hierarchy\n")
	end
	
	distances = Distances.pairwise(metric_instance, X, X, dims=1)
        clamp!(distances, tolerance, Inf)
        pairwise_self_to_infinity!(distances)
        return distances
    end

    function in_cluster_core_distance(distances::AbstractArray,
        d::Integer)::AbstractArray

        n = size(distances, 1)
        core_dists = (sum(distances .^ -d, dims=ndims(distances)) ./ (n - 1)) .^ (-1.0 / d)

        #manca filtraggio/clamping inverso

        return core_dists
    end

    function internal_objects(mutual_reach_distances::AbstractArray)::AbstractArray
        
        n = sqrt(length(mutual_reach_distances, 1))
        graph = SimpleWeightedGraphs.SimpleWeightedGraph(reshape(mutual_reach_distances, n, n))

        mst_edges = transpose(Graphs.kruskal_mst(graph))
        mst_matrix = deepcopy(mutual_reach_distances)


        for edge in mst_edges
            src, dest, w = Graphs.src(edge), Graphs.dst(edge), Graphs.weight(edge)
            mst_matrix[src, dest] = w
            mst_matrix[dest, src] = w
        end

        internal_nodes_i = findall(vec(count(>(0.0), mst_matrix, dims=1)) .> 1)
        internal_weights = get_subarray(mst_matrix, internal_nodes_i, nothing)


        if !isempty(internal_nodes_i)
            if length(internal_weights) > 1
                return (internal_nodes_i, internal_weights)
            else
                return (internal_nodes_i, mst_matrix)
            end
        elseif length(internal_weights) > 1
                return (range(size(mutual_reach_distances, 1), step=1), internal_weights)
        else
                return (range(size(mutual_reach_distances, 1), step=1), mst_matrix)
        end

    end

    function mutual_reachability_distances(mutual_distances::AbstractArray,
        d::Integer)::AbstractArray

        core_distances = in_cluster_core_distance(mutual_distances, d)
        cd_appo = transpose(core_distances)
        mrd = map(max, mutual_distances, core_distances, cd_appo)
        return (core_distances, mrd)
    end

    function get_subarray(arr, inds_a, inds_b)
        if isnothing(inds_a)
            return arr
        end

        actual_inds_b = isnothing(inds_b) ? inds_a : inds_b

        return transpose(arr[inds_a, actual_inds_b])
    end

    function density_sparseness(cluster_inds::AbstractArray,
        distances::AbstractArray{<:Number},
        d::Integer)

        core_distances, mutual_reach_distances = mutual_reachability_distances(distances, d)
        internal_node_inds, internal_edge_weights = internal_objects(mutual_reach_distances)

        cluster_density_sparseness = max(internal_edge_weights)
        internal_core_distances = core_distances[internal_node_inds]
        internal_node_inds = cluster_inds[internal_node_inds]

        return (cluster_density_sparseness, internal_core_distances, internal_node_inds)
    end

    function density_separation(cluster_i::Integer,
        cluster_j::Integer,
        distances::AbstractArray{<:Number},
        internal_core_distances_i::AbstractArray{<:Number},
        internal_core_distances_j::AbstractArray{<:Number})


        separation = max.(distances, internal_core_distances_i, transpose(internal_core_distances_j))

        if !isempty(separation)
            density_sep_btwn_clusters = min(separation)
        else
            density_sep_btwn_clusters = Inf
        end

        return (cluster_i, cluster_j, density_sep_btwn_clusters)
    end

    function dbcv(X::AbstractArray,
        y::AbstractVector;
        metric::AbstractString = "SqEuclidean",
        noise_id::Integer = -1,
        check_duplicates::Bool = true,
        sep_threshold = 1e-9,
        #enable_dynamic_precision::Bool = false,
        #enable_dynamic_precision ignored, that's the default in Julia!
        bits_ot_precision::Integer = 512, #approfondire se costruire tipo dato o usare fixed prec aritm.
    )::AbstractFloat

        n, d = size(X)

        if n != size(y, 1)
            throw(ArgumentError("Mismatch in input data (X) lenght and their clustering id assignments (y)"))
        end

        #clusters containing a single element can have that element regarded as noise.
        #reassign all noise to the noise_id cluster, by default id -1

        bool_keep_matrix = convert_singleton_clusters_to_noise!(y, noise_id)
        
        #filtering the AbstractArray does flattern it
        y = y[bool_keep_matrix]
        #keep whole colums, on true, bool_keep_matrix is not a flattened index for X!
        X = X[bool_keep_matrix, :]

        if isempty(y)
            return 0.0
        end

        y = dense_ranking(y)

        # forse meglio cambiare nome variabili per cluster_ids, ora è un rank non il vecchio cluster_id
        # ma saranno i miei nuovi identificatori di posizione per i cluster, tolto il rumore

        cluster_ids = unique(y)
        num_clusters::Integer = length(cluster_ids)
        cluster_sizes =  [count(==(id), y) for id in cluster_ids]

        if check_duplicates
            check_duplicated_samples(X, threshold=sep_threshold)
        end

        distances = pair_to_pair_distances(X, metric=metric, threshold=sep_threshold)

        # DSC: 'Density Sparseness of a Cluster' init
        dscs = zeros(size(cluster_ids))

        # DSC: 'Density Sparseness of a Cluster' init
        min_dspcs = fill(Inf, size(cluster_ids))

        # core distances of internal nodes
        internal_objects_per_cluster::Dict = Dict{Integer, AbstractArray{<:Integer}}()
        # insert with internal_objects_per_cls[key] = newvalue

        internal_core_distances_per_cluster::Dict = Dict{Integer, AbstractArray{<:Number}}()

        cluster_indexes = [findall(y .== cls_id) for cls_id in cluster_ids]

        #scegliere implementazione migliore multithreading
        #e divisione in sottomatrici/sottoproblemi

        Threads.@threads for i in eachindex(cluster_ids)
            subcls_indexes = cluster_indexes[i]
            dscs[i],
            internal_core_distances_per_cluster[i],
            internal_objects_per_cluster[i] =
            density_sparseness(subcls_indexes, get_subarray(distances,subcls_indexes, nothing), d)
        end

        number_cluster_pairs::Integer = fld((num_clusters*(num_clusters - 1)), 2)

        if number_cluster_pairs > 0
            tasks = [Threads.@spawn density_separation(
                    pair[1],
                    pair[2],
                    get_subarray(distances,
                        internal_objects_per_cluster[pair[1]],
                        internal_objects_per_cluster[pair[2]]),
                    internal_core_distances_per_cluster[pair[1]],
                    internal_core_distances_per_cluster[pair[2]]) 
                for pair in Combinatorics.combinations(cluster_ids, 2)]
            
            results = fetch.(tasks)

            for result in eachrow(results)
                cluster_a, cluster_b, density_sep = result
                min_dspcs[cluster_a] = min(min_dspcs[cluster_a], density_sep)
                min_dspcs[cluster_b] = min(min_dspcs[cluster_b], density_sep)
            end
        end

        # verificare se necessario equivalente di np.nan_to_num(min_dspcs, copy=False, posinf=1e12)

        base::AbstractFloat = sep_threshold/1e-3
        vcs::AbstractArray = (min_dspcs .- dscs) ./ (base .+ max.(min_dspcs, dscs))

        # verificare se necessario equivalente di np.nan_to_num(vcs, copy=False, nan=0.0)

        dbcv = sum(vcs .* cluster_sizes) / n
        return dbcv
    end
end
