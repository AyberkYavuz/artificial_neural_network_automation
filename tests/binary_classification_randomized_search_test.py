from sklearn.datasets import load_breast_cancer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD
from artificial_neural_network_model_automation.hyper_parameter_tuning import ANNRandomizedSearchConfig
from artificial_neural_network_model_automation.hyper_parameter_tuning import ANNRandomizedSearch


# loading data
data = load_breast_cancer()
features = data.data
target = data.target


# creating ANNClassificationRandomizedSearchConfig object
optimizer1 = Adam()
optimizer2 = Adam(learning_rate=0.01)
optimizer3 = Adam(learning_rate=0.02)
optimizer4 = SGD()
optimizer5 = SGD(learning_rate=0.02)

machine_learning_task = "binary"
scoring = "roc_auc"
neural_network_architecture_list = [
    [30, 40, 40, 1],
    [30, 30, 1],
    [30, 50, 1],
    [30, 70, 1],
    [30, 80, 1]
]
hidden_layers_activation_function_list = ["relu", "selu", "elu", "sigmoid"]
dropout_rate_list = [None, 0.0001, 0.001, 0.01, 0.02, 0.03, 0.04]
optimizer_list = [optimizer1, optimizer2, optimizer3, optimizer4, optimizer5]
metric_list = ["AUC", "Recall", "Precision"]
batch_size_list = [10, 20, 30, 40, 50]
epochs_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

neural_network_config_list_dict = {
    "machine_learning_task": machine_learning_task,
    "scoring": scoring,
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
ann_randomized_search.fit(features, target)

