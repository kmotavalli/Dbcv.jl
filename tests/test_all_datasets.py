#!/usr/bin/env python3

import subprocess

subprocess.run(["julia", "./InstallDeps.jl"])

default_params = {
    "samples": 500,
    "seed": 30,
    "eps": 0.25,
    "noise": 0.05,
    "factor": 0.5,
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
    )
]

for (testname, *custom_params) in test_parameters:
    params = default_params.copy()
    
    if custom_params:
        params.update(custom_params[0])

    subprocess.run(["python3", "./generalized_tester.py", "--testname", testname, "--numsamples", str(params["samples"]),
                    "--seed", str(params["seed"]), "--noiselevel", str(params["noise"]), "--eps", str(params["eps"]), "--factor", str(params["factor"])])
