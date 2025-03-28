# import net
import math
import numpy as np
import argparse
import sys
import os
from net import net
from skopt import gp_minimize
from skopt.space import Real
# from skopt.utils import use_named_args

# python -u "c:\Users\amish\Documents\open\DropoutUncertaintyExps\experiment.py" --dir bostonHousing
parser = argparse.ArgumentParser()


parser.add_argument('--dir', '-d', required=True,
                    help='Name of the UCI Dataset directory. Eg: bostonHousing')
parser.add_argument('--epochx', '-e', default=2, type=int,
                    help='Multiplier for the number of epochs for training.')
parser.add_argument('--hidden', '-nh', default=2, type=int,
                    help='Number of hidden layers for the neural net')

args = parser.parse_args()

data_directory = args.dir
epochs_multiplier = args.epochx
num_hidden_layers = args.hidden

sys.path.append('net/')


# Construct paths in a Windows-compatible way
base_results_path = os.path.join("UCI_Datasets", data_directory, "results")
base_data_path = os.path.join("UCI_Datasets", data_directory, "data")

result_files = {
    "validation_ll": os.path.join(base_results_path, f"validation_ll_{epochs_multiplier}_xepochs_{num_hidden_layers}_hidden_layers.txt"),
    "validation_rmse": os.path.join(base_results_path, f"validation_rmse_{epochs_multiplier}_xepochs_{num_hidden_layers}_hidden_layers.txt"),
    "validation_MC_rmse": os.path.join(base_results_path, f"validation_MC_rmse_{epochs_multiplier}_xepochs_{num_hidden_layers}_hidden_layers.txt"),
    "test_ll": os.path.join(base_results_path, f"test_ll_{epochs_multiplier}_xepochs_{num_hidden_layers}_hidden_layers.txt"),
    "test_tau": os.path.join(base_results_path, f"test_tau_{epochs_multiplier}_xepochs_{num_hidden_layers}_hidden_layers.txt"),
    "test_rmse": os.path.join(base_results_path, f"test_rmse_{epochs_multiplier}_xepochs_{num_hidden_layers}_hidden_layers.txt"),
    "test_MC_rmse": os.path.join(base_results_path, f"test_MC_rmse_{epochs_multiplier}_xepochs_{num_hidden_layers}_hidden_layers.txt"),
    "test_log": os.path.join(base_results_path, f"log_{epochs_multiplier}_xepochs_{num_hidden_layers}_hidden_layers.txt"),
}
# dictionary to map file paths with usable keys
data_files = {
    "dropout_rates": os.path.join(base_data_path, "dropout_rates.txt"),
    "tau_values": os.path.join(base_data_path, "tau_values.txt"),
    "data": os.path.join(base_data_path, "data.txt"),
    "hidden_units": os.path.join(base_data_path, "n_hidden.txt"),
    "epochs": os.path.join(base_data_path, "n_epochs.txt"),
    "index_features": os.path.join(base_data_path, "index_features.txt"),
    "index_target": os.path.join(base_data_path, "index_target.txt"),
    "n_splits": os.path.join(base_data_path, "n_splits.txt"),
}


def get_index_train_test_path(split_num, train=True):
    """Generate path for train/test split"""
    filename = f"index_train_{split_num}.txt" if train else f"index_test_{split_num}.txt"
    return os.path.join(base_data_path, filename)  
# saving the index for dataset splits


print("Removing existing result files...")
for file in result_files.values():
    if os.path.exists(file):
        os.remove(file)
print("Result files removed.")

# Fix random seed
np.random.seed(1)

print("Loading data and other hyperparameters...")

# Load data


# Load data
data = np.loadtxt(data_files["data"])
n_hidden = int(np.loadtxt(data_files["hidden_units"]))
n_epochs = int(np.loadtxt(data_files["epochs"]))
index_features = np.loadtxt(data_files["index_features"])
index_target = int(np.loadtxt(data_files["index_target"]))
X = data[:, [int(i) for i in index_features.tolist()]]
y = data[:, index_target]
n_splits = int(np.loadtxt(data_files["n_splits"]))

print("Data loading completed.")

def objective(dropout_tau):
    dropout_rate, tau = dropout_tau
    network = net.net(X_train, y_train, ([n_hidden] * num_hidden_layers),
                      normalize=True, n_epochs=int(n_epochs * epochs_multiplier),
                      tau=tau, dropout=dropout_rate)
    error, MC_error, ll = network.predict(X_validation, y_validation)
    return -ll  # Negative log-likelihood (to minimize)

for split in range(n_splits):
    index_train = np.loadtxt(get_index_train_test_path(split, train=True))
    index_test = np.loadtxt(get_index_train_test_path(split, train=False))
    X_train = X[[int(i) for i in index_train.tolist()]]
    y_train = y[[int(i) for i in index_train.tolist()]]
    X_test = X[[int(i) for i in index_test.tolist()]]
    y_test = y[[int(i) for i in index_test.tolist()]]
    num_training_examples = int(0.8 * X_train.shape[0])
    X_validation = X_train[num_training_examples:, :]
    y_validation = y_train[num_training_examples:]
    X_train = X_train[:num_training_examples, :]
    y_train = y_train[:num_training_examples]
    
    # Bayesian Optimization Search Space
    search_space = [Real(0.001, 0.2, name='dropout_rate'), Real(0.01, 0.2, name='tau')]
    res = gp_minimize(objective, search_space, n_calls=15, random_state=42)
    best_dropout, best_tau = res.x
    print("The best droupout is: ", best_dropout," and tau value is: " ,best_tau)
    
    best_network = net.net(X_train, y_train, ([n_hidden] * num_hidden_layers),
                           normalize=True, n_epochs=int(n_epochs * epochs_multiplier),
                           tau=best_tau, dropout=best_dropout)
    error, MC_error, ll = best_network.predict(X_test, y_test)
    
    # Save results
    with open(result_files["test_rmse"], "a") as myfile:
        myfile.write(f'{error}\n')
    with open(result_files["test_MC_rmse"], "a") as myfile:
        myfile.write(f'{MC_error}\n')
    with open(result_files["test_ll"], "a") as myfile:
        myfile.write(f'{ll}\n')
    with open(result_files["test_tau"], "a") as myfile:
        myfile.write(f'{best_tau}\n')
    with open(result_files["test_log"], "a") as myfile:
        myfile.write(f'Split {split}: Best tau: {best_tau}, Best dropout: {best_dropout}, LL: {ll}\n')

print("Training completed!")
