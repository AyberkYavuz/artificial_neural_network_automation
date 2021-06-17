from sklearn.datasets import load_iris
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD
from artificial_neural_network_model_automation.hyper_parameter_tuning import ANNRandomizedSearchConfig
from artificial_neural_network_model_automation.hyper_parameter_tuning import ANNRandomizedSearch

# loading data
data = load_iris()
features = data.data
target = data.target


# creating ANNClassificationRandomizedSearchConfig object
machine_learning_task = "multiclass"
neural_network_architecture_list = [
    [4, 9, 12, 9, 3],
    [4, 15, 9, 3],
    [4, 25, 3],
    [4, 10, 10, 10, 10, 3],
    [4, 8, 16, 8, 3],
    [4, 16, 16, 3],
    [4, 4, 4, 4, 4, 3],
    [4, 50, 25, 3],
    [4, 15, 15, 16, 3],
    [4, 12, 12, 12, 3]
]
hidden_layers_activation_function_list = ["relu", "sigmoid", "tanh", "selu", "elu"]
dropout_rate_list = [None, 0.0001, 0.001, 0.01, 0.02, 0.03, 0.04]

optimizer1 = Adam()
optimizer2 = Adam(learning_rate=0.01)
optimizer3 = Adam(learning_rate=0.02)
optimizer4 = SGD()
optimizer5 = SGD(learning_rate=0.02)
optimizer_list = [optimizer1, optimizer2, optimizer3, optimizer4, optimizer5]

metric_list = ["AUC", "Recall", "Precision"]
batch_size_list = [10, 20, 30, 40, 50]
epochs_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

neural_network_config_list_dict = {
    "machine_learning_task": machine_learning_task,
    "neural_network_architecture_list": neural_network_architecture_list,
    "hidden_layers_activation_function_list": hidden_layers_activation_function_list,
    "dropout_rate_list": dropout_rate_list,
    "optimizer_list": optimizer_list,
    "metric_list": metric_list,
    "batch_size_list": batch_size_list,
    "epochs_list": epochs_list
}

ann_randomized_search_config = ANNRandomizedSearchConfig(neural_network_config_list_dict)

# create artificial neural network randomized search object
ann_randomized_search = ANNRandomizedSearch(ann_randomized_search_config, n_iter=100, n_jobs=-1)
# perform randomized search
ann_randomized_search.fit(features, target, target_categories=[0, 1, 2])
