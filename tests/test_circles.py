#!/usr/bin/env python3

import time, sys, os, subprocess

import numpy as np

#richiede installazione sklearn e numpy
import felsiq_dbcv as dbcv #felsiq implementation
from sklearn import cluster, datasets
from sklearn.neighbors import kneighbors_graph

def main():
    now = time.localtime()

    tstringdir = str(now.tm_year) + "-" + str(now.tm_mon) + "-" + str(now.tm_mday)
    tstringfile = str(now.tm_year) + "-" + str(now.tm_mon) + "-" + str(now.tm_mday) + "_" + str(now.tm_hour) + str(now.tm_min)

    n_samples = 500
    seed = 30
    noise=0.05

    if len(sys.argv) > 1:
        n_samples = sys.argv[1]
    if len(sys.argv) > 2:
        noise = sys.argv[2]
    if len(sys.argv) > 3:
        seed = sys.argv[3]

    noisy_circles = datasets.make_circles(
        n_samples=n_samples, factor=0.5, noise=noise, random_state=seed
    )

    print("Datapoints: \n")
    print(noisy_circles[0])


    dir_path = "../data/tests/" + "circles_" + tstringdir
    dataset_path = dir_path + "/" + "circles_dataset_" + tstringfile + ".csv"
    classification_path = dir_path + "/" + "circles_classification_" + tstringfile + ".csv"


    try:
        os.makedirs(dir_path)
    except:
        print(dir_path + " already exists, saving dataset")

    np.savetxt(dataset_path , noisy_circles[0], delimiter= ",")

    plot_num = 1

    dbscan = cluster.DBSCAN(eps=0.25).fit(noisy_circles[0])
    classification = np.array(dbscan.labels_, dtype=np.int32)
    print("\nClassification: \n")
    print(dbscan.labels_)

    np.savetxt(classification_path, classification, fmt='%d', delimiter= ",")

    #dbcv score calculations across implementations

    py_score = dbcv.dbcv(noisy_circles[0], classification)

    print("\nDBCV score with the py felsiq implementation: " + str(py_score) + "\n")

    output = subprocess.run(["julia", "./Tester.jl", dataset_path, classification_path], stdout = subprocess.PIPE, universal_newlines = True)    
    julia_score = float(output.stdout)

    print("DBCV score with the Julia implementation: " + str(julia_score) + "\n")

    difference = py_score - julia_score

    print("Difference: " + str(difference) + "\nSaving on " + dir_path + "/circles_result.txt\n")

    with open(dir_path + "/circles_result.txt", 'w') as out:
        out.write("DBCV Scores for " + "circles_dataset_" + tstringfile + ".csv\n")
        out.write("Python: " + str(py_score) + "\n")
        out.write("Julia: " + str(julia_score) + "\n\n")
        out.write("Difference: " + str (difference) + "\n")
        out.close()

if __name__ == '__main__':
    main()
