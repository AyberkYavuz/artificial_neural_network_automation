from pandas import read_csv
from sklearn.preprocessing import LabelEncoder
from helper.data_path_handler import get_data_path
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD
from artificial_neural_network_model_automation.hyper_parameter_tuning import ANNClassificationRandomizedSearchConfig
from artificial_neural_network_model_automation.hyper_parameter_tuning import ANNClassificationRandomizedSearch

data_name = "sonar.csv"
# please define your os_type (mac, windows, linux)
data_path = get_data_path(data_name, directory_name="data", os_type="mac")


# load dataset
dataframe = read_csv(data_path, header=None)
dataset = dataframe.values

# split into input (X) and output (Y) variables
X = dataset[:, 0:60].astype(float)
print(X.shape)
Y = dataset[:, 60]
print(Y.shape)

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)

# creating ANNClassificationRandomizedSearchConfig object
optimizer1 = Adam()
optimizer2 = Adam(learning_rate=0.01)
optimizer3 = Adam(learning_rate=0.02)
optimizer4 = SGD()
optimizer5 = SGD(learning_rate=0.02)

classification_type = "binary"
scoring = "roc_auc"
neural_network_architecture_list = [
    [60, 65, 65, 1],
    [60, 70, 70, 1],
    [60, 80, 1],
    [60, 90, 1],
    [60, 85, 1],
    [60, 85, 85, 1],
    [60, 85, 85, 85, 1],
    [60, 75, 75, 75, 1],
    [60, 60, 1],
    [60, 60, 60, 1],
    [60, 60, 60, 60, 1],
    [60, 60, 60, 60, 60, 1],
    [60, 80, 70, 1],
    [60, 90, 80, 70, 1],
]
hidden_layers_activation_function_list = ["relu", "sigmoid", "tanh", "selu", "elu"]
dropout_rate_list = [None, 0.0001, 0.001, 0.01, 0.02, 0.03, 0.04]
optimizer_list = [optimizer1, optimizer2, optimizer3, optimizer4, optimizer5]
metric_list = ["AUC", "Recall", "Precision"]
batch_size_list = [10, 20, 30, 40, 50]
epochs_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

neural_network_config_list_dict = {
    "classification_type": classification_type,
    "scoring": scoring,
    "neural_network_architecture_list": neural_network_architecture_list,
    "hidden_layers_activation_function_list": hidden_layers_activation_function_list,
    "dropout_rate_list": dropout_rate_list,
    "optimizer_list": optimizer_list,
    "metric_list": metric_list,
    "batch_size_list": batch_size_list,
    "epochs_list": epochs_list
}

ann_classification_randomized_search_config = ANNClassificationRandomizedSearchConfig(neural_network_config_list_dict)


ann_classification_randomized_search = ANNClassificationRandomizedSearch(ann_classification_randomized_search_config, n_jobs=-1)
ann_classification_randomized_search.fit(X, encoded_Y)  # Method Execution Time : 0.12 minutes

# parallelism test
ann_classification_randomized_search_2 = ANNClassificationRandomizedSearch(ann_classification_randomized_search_config,
                                                                           n_iter=100, n_jobs=-1)
ann_classification_randomized_search_2.fit(X, encoded_Y)  # Method Execution Time : 0.39 minutes
