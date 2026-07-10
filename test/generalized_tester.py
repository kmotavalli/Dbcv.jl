#!/usr/bin/env python3

import time, sys, os, subprocess, argparse, multiprocessing
from inspect import getsourcefile
import numpy as np

#requires installing scipy, scikit-learn and numpy
import felsiq_dbcv as dbcv #felsiq implementation
from sklearn import cluster, datasets, preprocessing

def main():
	now = time.localtime()
	ncores = multiprocessing.cpu_count()
	is_medical = False
	test_scripts_dir = os.path.dirname(os.path.abspath(getsourcefile(lambda:0)))
	results_file = ""



	tstringdir = str(now.tm_year) + "-" + str(now.tm_mon) + "-" + str(now.tm_mday)
	tstringfile = str(now.tm_year) + "-" + str(now.tm_mon) + "-" + str(now.tm_mday) + "_" + str(now.tm_hour) + str(now.tm_min)


	parser = argparse.ArgumentParser()
	parser.add_argument('--testname', type=str, default="circles")
	parser.add_argument('--numsamples', type=int, default=500)
	parser.add_argument('--seed', type=int, default=30)
	parser.add_argument('--noiselevel', type=float, default=0.05)
	parser.add_argument('--eps', type=float, default=0.25)
	parser.add_argument('--factor', type=float, default=0.5)
	parser.add_argument('--shape', type=int, nargs="+", default=None)
	parser.add_argument('--n_clusters', type=int, default=None)
	parser.add_argument('--n_features', type=int, default=None)
	parser.add_argument('--n_dim', type=int, default=None)
	args = parser.parse_args()


	if args.testname.endswith(".csv") and os.path.exists(args.testname):
		tempload = np.loadtxt(args.testname , skiprows=1, delimiter= ",")
		scaler_inst = preprocessing.StandardScaler()
		dataset = scaler_inst.fit_transform(tempload)
		is_medical = True
		print("Running test for dataset " + str(args.testname))
		print("ARGS: " + str(args))


	else:
		print("Running test for SkLearn dataset make_" + str(args.testname))
		print("ARGS: " + str(args))
		ds_method = getattr(datasets, 'make_' + args.testname)
    
		match args.testname:
			case "circles":
				dataset = ds_method(
					n_samples=args.numsamples, factor=args.factor, noise=args.noiselevel, random_state=args.seed)[0]
			case "moons":
				dataset = ds_method(
					n_samples=args.numsamples, noise=args.noiselevel)[0]
			case "blobs":
				dataset = ds_method(
					n_samples=args.numsamples, random_state=args.seed)[0]
			case "biclusters" | "checkerboard":
				dataset = ds_method(
					(args.shape[0], args.shape[1]), args.n_clusters, noise=args.noiselevel, random_state=args.seed)[0]
			case "low_rank_matrix":
				dataset = ds_method(
					n_samples=args.numsamples, n_features=args.n_features, random_state=args.seed)
			case "s_curve":
				dataset = ds_method(
					n_samples=args.numsamples, noise=args.noiselevel, random_state=args.seed)[0]
			case "spd_matrix":
				dataset = ds_method(n_dim=args.n_dim, random_state=args.seed) #could also pass alpha=, etc
			case "sparse_coded_signal":
				_, _, dataset = ds_method(n_samples=args.numsamples, n_features=args.n_features, n_components=5, n_nonzero_coefs=2, random_state=args.seed)
				print(dataset)
			case "sparse_spd_matrix":
				dataset = ds_method(n_dim=args.n_dim, random_state=args.seed) #could also pass alpha=, etc
			case _:
				dataset = ds_method(n_samples=args.numsamples, random_state=args.seed)[0]

		print("Datapoints: \n")
		print(dataset)
	
	
	if is_medical:
		dataset_path = args.testname
		dir_path = os.path.join(test_scripts_dir, "..", "data", "tests", "medical_" + os.path.basename(args.testname).split(".csv")[0] + "_" + tstringdir)
		classification_path = os.path.join(dir_path, os.path.basename(args.testname).split(".csv")[0] + "_classification_" + tstringfile + ".csv")
		results_file = os.path.join(dir_path, os.path.basename(args.testname).split(".csv")[0] + "_result_ " + tstringfile + ".txt")
		normalized_ds_path = os.path.join(dir_path, os.path.basename(args.testname).split(".csv")[0] + "_normalizeddataset_" + tstringfile + ".csv")

	else:
		dir_path = os.path.join(test_scripts_dir, "..", "data", "tests", args.testname + "_" + tstringdir)
		dataset_path = os.path.join(dir_path , args.testname + "_dataset_"  + tstringfile + ".csv")
		classification_path = os.path.join(dir_path, args.testname + "_classification_" + tstringfile + ".csv")
		results_file = os.path.join(dir_path, args.testname + "_result_ " + tstringfile + ".txt")


	try:
		os.makedirs(dir_path)
	except:
		print(dir_path + " already exists, saving there")

	if not is_medical:
		np.savetxt(dataset_path , dataset, delimiter= ",")
	else:
		np.savetxt(normalized_ds_path, dataset, delimiter=",")

	dbscan = cluster.DBSCAN(eps=args.eps).fit(dataset)
	classification = np.array(dbscan.labels_, dtype=np.int32)
	print("\nClassification: \n")
	print(dbscan.labels_)

	np.savetxt(classification_path, classification, fmt='%d', delimiter= ",")

    #dbcv score calculations across implementations

	py_score = dbcv.dbcv(dataset, classification, use_original_mst_implementation=False)

	print("\nDBCV score with the py felsiq implementation: " + str(py_score) + "\n")

	if is_medical:
		output = subprocess.run(["julia", f'-t {ncores}', os.path.join(test_scripts_dir, "Tester.jl"), normalized_ds_path, classification_path], stdout = subprocess.PIPE, universal_newlines = True)
	else:
		output = subprocess.run(["julia", f'-t {ncores}', os.path.join(test_scripts_dir, "Tester.jl"),  dataset_path, classification_path], stdout = subprocess.PIPE, universal_newlines = True)

	print(output.args)

	julia_score = float(output.stdout)

	print("DBCV score with the Julia implementation: " + str(julia_score) + "\n")

	difference = py_score - julia_score

	print("Difference: " + str(difference) + "\nSaving on " + results_file + "\n")

	with open(results_file, 'w') as out:
		out.write("DBCV Scores for " + dataset_path + "\n")
		out.write("Python:  " + str(py_score) + "\n")
		out.write("Julia :  " + str(julia_score) + "\n\n")
		out.write("Difference: " + str (difference) + "\n")
		out.close()

if __name__ == '__main__':
	main()
