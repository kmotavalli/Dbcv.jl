#!/usr/bin/env python3

import time, sys, os, subprocess, argparse, multiprocessing

import numpy as np

#richiede installazione sklearn e numpy
import felsiq_dbcv as dbcv #felsiq implementation
from sklearn import cluster, datasets

def main():
    now = time.localtime()
    ncores = multiprocessing.cpu_count()

    tstringdir = str(now.tm_year) + "-" + str(now.tm_mon) + "-" + str(now.tm_mday)
    tstringfile = str(now.tm_year) + "-" + str(now.tm_mon) + "-" + str(now.tm_mday) + "_" + str(now.tm_hour) + str(now.tm_min)


    parser = argparse.ArgumentParser()
    parser.add_argument('--testname', type=str, default="circles")
    parser.add_argument('--numsamples', type=int, default=500)
    parser.add_argument('--seed', type=int, default=30)
    parser.add_argument('--noiselevel', type=float, default=0.05)
    parser.add_argument('--eps', type=float, default=0.25)
    parser.add_argument('--factor', type=float, default=0.5)

    args = parser.parse_args()

    ds_method = getattr(datasets, 'make_' + args.testname)

    match args.testname:
        case "circles":
            dataset = ds_method(
            n_samples=args.numsamples, factor=args.factor, noise=args.noiselevel, random_state=args.seed)
        case "moons":
            dataset = ds_method(
            n_samples=args.numsamples, noise=args.noiselevel)
        case "blobs":
            dataset = ds_method(
            n_samples=args.numsamples, random_state=args.seed)

            

    print("Datapoints: \n")
    print(dataset[0])


    dir_path = "../data/tests/" + args.testname + "_" + tstringdir
    dataset_path = dir_path + "/" + args.testname + "_dataset_"  + tstringfile + ".csv"
    classification_path = dir_path + "/" + args.testname + "_classification_" + tstringfile + ".csv"


    try:
        os.makedirs(dir_path)
    except:
        print(dir_path + " already exists, saving dataset")

    np.savetxt(dataset_path , dataset[0], delimiter= ",")


    dbscan = cluster.DBSCAN(eps=args.eps).fit(dataset[0])
    classification = np.array(dbscan.labels_, dtype=np.int32)
    print("\nClassification: \n")
    print(dbscan.labels_)

    np.savetxt(classification_path, classification, fmt='%d', delimiter= ",")

    #dbcv score calculations across implementations

    py_score = dbcv.dbcv(dataset[0], classification)

    print("\nDBCV score with the py felsiq implementation: " + str(py_score) + "\n")

    output = subprocess.run(["julia", f'-t {ncores}', "./Tester.jl", dataset_path, classification_path], stdout = subprocess.PIPE, universal_newlines = True)    
    julia_score = float(output.stdout)

    print("DBCV score with the Julia implementation: " + str(julia_score) + "\n")

    difference = py_score - julia_score

    print("Difference: " + str(difference) + "\nSaving on " + dir_path + "/" + args.testname + "_result_" + tstringfile + ".txt\n")

    with open(dir_path + "/" + args.testname + "_result_ " + tstringfile + ".txt", 'w') as out:
        out.write("DBCV Scores for " + args.testname + "_dataset_" + tstringfile + ".csv\n")
        out.write("Python: " + str(py_score) + "\n")
        out.write("Julia: " + str(julia_score) + "\n\n")
        out.write("Difference: " + str (difference) + "\n")
        out.close()

if __name__ == '__main__':
    main()
