#!/usr/bin/env python3

import subprocess, random

subprocess.run(["julia", "./InstallDeps.jl"])

default_params = {
	"samples": 500,
	"seed": 30,
	"eps": 0.25,
	"noise": 0.05,
	"factor": 0.5,
	"n_dim": None,
	"n_features": 1,
}

test_parameters = [
	(
		"circles",
		{"samples": 500,
		"seed": 30,
		"eps": 0.25,
		"noise": 0.05},
	),
	(
		"moons",
		{"samples" : 400},
	),
	(
		"blobs",
	),
    (
		"biclusters",
		{
			"shape": (800, 800),
			"n_clusters": random.randrange(80),
		}
	),
	(
		"checkerboard",
		{
			"shape": (800, 800),
			"n_clusters": random.randrange(80),
		}
	),
	(
		"classification",
	),
	(
		"friedman1",
	),
	(
		"friedman2",
	),
	(
		"friedman3",
	),
	(
		"gaussian_quantiles",
	),
	(
		"hastie_10_2",
	),
	(
		"low_rank_matrix",
		{
			"n_features": random.randrange(10),
		}
	),
	(
		"multilabel_classification",
	),
	(
		"regression",
	),
	(
		"s_curve",
	),
	(
		"sparse_coded_signal",
		{"n_features": 1},
	),
	(
		"sparse_spd_matrix",
		{"n_dim": 5},
	),
	(
		"sparse_uncorrelated",
	),
	(
		"spd_matrix",
		{"n_dim": 5},
	),
	(
		"swiss_roll",
	),
]

for (testname, *custom_params) in test_parameters:
	params = default_params.copy()
	if custom_params:
		params.update(custom_params[0])
	match testname:
		case "low_rank_matrix":
			subprocess.run(["python3", "./generalized_tester_prim.py", "--testname", testname, "--numsamples", str(params["samples"]),
				"--eps", str(params["eps"]), "--n_features", str(params["n_features"])])
		case "biclusters" | "checkerboard":
			subprocess.run(["python3", "./generalized_tester_prim.py", "--testname", testname, "--shape", str(params["shape"][0]), str(params["shape"][1]), "--n_clusters", str(params["n_clusters"]), "--numsamples", str(params["samples"]),
				"--seed", str(params["seed"]), "--noiselevel", str(params["noise"]), "--eps", str(params["eps"]), "--factor", str(params["factor"])])
		case _:
			if params["n_dim"] is not None:
				subprocess.run(["python3", "./generalized_tester_prim.py", "--testname", testname, "--numsamples", str(params["samples"]),
					"--seed", str(params["seed"]),  "--n_features", str(params["n_features"]), "--noiselevel", str(params["noise"]), "--n_dim", str(params["n_dim"]), "--eps", str(params["eps"]), "--factor", str(params["factor"])])
			else:
				subprocess.run(["python3", "./generalized_tester_prim.py", "--testname", testname, "--numsamples", str(params["samples"]),
					"--seed", str(params["seed"]),  "--n_features", str(params["n_features"]), "--noiselevel", str(params["noise"]), "--eps", str(params["eps"]), "--factor", str(params["factor"])])

