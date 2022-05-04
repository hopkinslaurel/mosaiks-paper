import os
import pickle
from pathlib import Path

from mosaiks import transforms
from mosaiks.utils import OVERWRITE_EXCEPTION
from mosaiks.utils.imports import *
from sklearn.metrics import r2_score
#from threadpoolctl import threadpool_limits

num_threads = None

subset_n = slice(None)
subset_feat = slice(None)

overwrite = None
fixed_lambda = False

labels_to_run = ["treecover"] #"all"

if overwrite is None:
	overwrite = os.getenv("MOSAIKS_OVERWRITE", False)
if labels_to_run == "all":
	labels_to_run = c.app_order

"""if num_threads is not None:
	threadpool_limits(num_threads)
	os.environ["NUMBA_NUM_THREADS"] = str(num_threads)"""

solver = solve.ridge_regression

# define output location
save_patt = join(
	"{save_dir}",
	"outcomes_{{reg_type}}_obsAndPred_{label}_{variable}_CONTUS_16_640_{sampling}_"
	f"{c.sampling['n_samples']}_{c.sampling['seed']}_random_features_{c.features['random']['patch_size']}_"
	f"{c.features['random']['seed']}{{subset}}.data",
)

# load random feature data
X = {}
latlons = {}

X["UAR"], latlons["UAR"] = io.get_X_latlon(c, "UAR")
X["POP"], latlons["POP"] = io.get_X_latlon(c, "POP")

# run regressions
results_dict = {}
results_dict_test = {}
for label in labels_to_run:

	print("*** Running regressions for: {}".format(label))

	## Set some label-specific variables
	c = io.get_filepaths(c, label)
	c_app = getattr(c, label)
	sampling_type = c_app["sampling"]  # UAR or POP
	this_save_patt = save_patt.format(
		subset="",
		save_dir=c.fig_dir_prim,
		label=label,
		variable=c_app["variable"],
		sampling=c_app["sampling"],
	)

	# decide wehether to just test the best lambda(s) or all of them
	if fixed_lambda:
		best_lambda_fpath = this_save_patt.format(reg_type="scatter", subset="")
	else:
		best_lambda_fpath = None
	this_lambdas = io.get_lambdas(c, label, best_lambda_fpath=best_lambda_fpath)

	# determine bounds of predictions
	if c_app["logged"]:
		bounds = np.array([c_app["us_bounds_log_pred"]])
	else:
		bounds = np.array([c_app["us_bounds_pred"]])

	## Get save path
	if (subset_n != slice(None)) or (subset_feat != slice(None)):
		subset_str = "_subset"
	else:
		subset_str = ""
	save_path_validation = this_save_patt.format(reg_type="scatter", subset=subset_str)
	save_path_test = this_save_patt.format(reg_type="testset", subset=subset_str)

	if (not overwrite) and (
		os.path.isfile(save_path_validation) or os.path.isfile(save_path_test)
	):
		raise OVERWRITE_EXCEPTION

	## get X, Y, latlon values of training data
	(
		this_X,
		this_X_test,
		this_Y,
		this_Y_test,
		this_latlons,
		this_latlons_test,
		this_split
	) = parse.merge_dropna_transform_split_train_test(
		c, label, X[sampling_type], latlons[sampling_type]
	)

	## subset
	"""this_X = this_X[subset_n, subset_feat]  # w/subset_n=None, shape = (80000, 8192)
	this_X_test = this_X_test[:, subset_feat]
	this_Y = this_Y[subset_n]
	this_latlons = this_latlons[subset_n]"""

	## Train model using ridge regression and 5-fold cross-valiation
	## (number of folds can be adjusted using the argument n_folds)
	print("Training model...")
	import time

	st_train = time.time()
	kfold_results = solve.kfold_solve(
		this_X,
		this_Y,
		this_split,
		solve_function=solver,
		num_folds=c.ml_model["n_folds"],
		return_model=True,
		lambdas=this_lambdas,
		return_preds=True,
		svd_solve=False,
		clip_bounds=bounds,
	)
	print("")

	# get timing
	training_time = time.time() - st_train
	print("Training time:", training_time)
	print(kfold_results.keys())

	## Store the metrics and the predictions from the best performing model
	best_lambda_idx, best_metrics, best_preds = ir.interpret_kfold_results(
		kfold_results, "r2_score", hps=[("lambdas", c_app["lambdas"])]
	)
	best_lambda = this_lambdas[best_lambda_idx]

	## combine out-of-sample predictions over folds
	preds = np.vstack([solve.y_to_matrix(i) for i in best_preds.squeeze()]).squeeze()
	truth = np.vstack(
		[solve.y_to_matrix(i) for i in kfold_results["y_true_test"].squeeze()]
	).squeeze()

	# get latlons in same shuffled, cross-validated order
	"""ll = this_latlons[
		np.hstack([test for train, test in kfold_results["cv"].split(this_latlons)])
	]"""
	print("best lambdas")
	print(len(truth))
	print(len(kfold_results["lon_test"]))
	data = {
		"truth": truth,
		"preds": preds,
		"lon": kfold_results["lon_test"],  #ll[:, 1],
		"lat": kfold_results["lat_test"],  #ll[:, 0],
		"best_lambda": best_lambda,
	}

	## save validation set predictions
	print("Saving validation set results to {}".format(save_path_validation))
	with open(save_path_validation, "wb") as f:
		pickle.dump(data, f)
	results_dict = r2_score(truth, preds)

	## Get test set predictions
	st_test = time.time()
	holdout_results = solve.single_solve(
		this_X,
		this_X_test,
		this_Y,
		this_Y_test,
		lambdas=best_lambda,
		svd_solve=False,
		return_preds=True,
		return_model=False,
		clip_bounds=bounds,
	)

	# get timing
	test_time = time.time() - st_test
	print("Test set training time:", test_time)

	## Save test set predictions
	ll = this_latlons_test
	data = {
		"truth": holdout_results["y_true_test"],
		"preds": holdout_results["y_pred_test"][0][0][0],
		"lon": kfold_results["lon_test"],  #ll[:, 1],
		"lat": kfold_results["lat_test"],  #ll[:, 0],
	}

	print("Saving test set results to {}".format(save_path_test))
	with open(save_path_test, "wb") as f:
		pickle.dump(data, f)

	## Store the R2
	results_dict_test[label] = holdout_results["metrics_test"][0][0][0]["r2_score"]
	print("Full reg time", time.time() - st_train)
