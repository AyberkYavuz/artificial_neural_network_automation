from sklearn.datasets import load_boston
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import RMSprop
from artificial_neural_network_model_automation.hyper_parameter_tuning import ANNRandomizedSearchConfig
from artificial_neural_network_model_automation.hyper_parameter_tuning import ANNRandomizedSearch

# loading data
data = load_boston()
features = data.data
target = data.target

# creating ANNRandomizedSearchConfig object
machine_learning_task = "regression"
neural_network_architecture_list = [
    [13, 20, 20, 1],
    [13, 40, 1],
    [13, 20, 20, 20, 1],
    [13, 20, 30, 20, 1],
    [13, 20, 25, 20, 1],
    [13, 25, 25, 25, 1],
    [13, 25, 1],
    [13, 50, 1],
    [13, 50, 50, 1]
]
hidden_layers_activation_function_list = ["relu", "selu", "elu"]
dropout_rate_list = [None, 0.0001, 0.001, 0.01, 0.02, 0.03, 0.04]

optimizer1 = Adam()
optimizer2 = Adam(learning_rate=0.01)
optimizer3 = Adam(learning_rate=0.02)
optimizer4 = RMSprop()
optimizer5 = RMSprop(learning_rate=0.01)
optimizer6 = RMSprop(learning_rate=0.02)
optimizer_list = [optimizer1, optimizer2, optimizer3, optimizer4, optimizer5, optimizer6]

metric_list = ["mean_squared_error", "mean_absolute_error", "mean_absolute_percentage_error",
               "mean_squared_logarithmic_error", "cosine_similarity", "logcosh"]
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
ann_randomized_search.fit(features, target)
