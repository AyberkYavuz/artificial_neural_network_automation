# artificial_neural_network_automation
This repository is for automating artificial neural network model creation with tabular data using Keras framework.

## Project Structure
```bash
artificial_neural_network_automation/
├── LICENSE
├── README.md
├── setup.py
├── artificial_neural_network_model_automation
│   ├── __init__.py       
│   ├── artificial_neural_network_handler.py
│   └── hyper_parameter_tuning.py
├── helper
│   ├── __init__.py
│   ├── classification_helper.py
│   ├── data_path_handler.py
│   ├── decorators.py
│   ├── helper.py
│   ├── make_keras_pickable.py
│   └── regression_helper.py
└── tests
    ├── __init__.py
    ├── binary_classification_randomized_search_test.py
    ├── binary_classification_test.py
    ├── multi_class_classification_randomized_search_test.py
    ├── multi_class_classification_test.py
    ├── regression_randomized_search_test.py
    └── regression_test.py
```

## Installation
There are 2 ways to install this python package.

### Installation from Github Releases
First, you should check the releases in order to install latest release.
Please look at [Releases](https://github.com/AyberkYavuz/artificial_neural_network_automation/releases).

After checking releases you can install this python package from available tar.gz files to your platform.

Example Installation
```bash
python3 -m pip install https://github.com/AyberkYavuz/artificial_neural_network_automation/archive/refs/tags/0.0.1.tar.gz
```

## Usage
This repository supports binary classification, multi-class classification and regression.

### Binary Classification Usage Example
```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from artificial_neural_network_model_automation.artificial_neural_network_handler import ArtificialNeuralNetworkHandlerConfig
from artificial_neural_network_model_automation.artificial_neural_network_handler import ArtificialNeuralNetworkHandler
from sklearn.metrics import classification_report

# loading data
data = load_breast_cancer()
features = data.data
target = data.target

# splitting data
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.33, random_state=42)

# designing neural network

neural_network_config = {"machine_learning_task": "binary",
                         "neural_network_architecture": [30, 40, 40, 1],
                         "hidden_layers_activation_function": "relu",
                         "optimizer": "adam",
                         "metric": "Recall",
                         "batch_size": 10,
                         "epochs": 50}

# create artificial neural network configuration object
ann_handler_config = ArtificialNeuralNetworkHandlerConfig(neural_network_config)

# create artificial neural network handler object
ann_handler = ArtificialNeuralNetworkHandler(ann_handler_config)

# training designed neural network
ann_handler.train_neural_network(X_train, y_train)
# making predictions
y_pred = ann_handler.get_predictions(X_test)
# classification report
print(classification_report(y_test, y_pred))
```

### Randomized Search for Binary Classification Usage Example
```python
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
```

### Multi-class Classification Usage Example
```python
from sklearn.datasets import load_iris
from tensorflow.keras.optimizers import SGD
from artificial_neural_network_model_automation.artificial_neural_network_handler import ArtificialNeuralNetworkHandlerConfig
from artificial_neural_network_model_automation.artificial_neural_network_handler import ArtificialNeuralNetworkHandler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.utils import np_utils

# loading data
data = load_iris()
features = data.data
target = data.target

# splitting data
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.33, random_state=42)

# dummy target transformation
dummy_y_train = np_utils.to_categorical(y_train)

# designing neural network
optimizer = SGD(learning_rate=0.01)
neural_network_config = {"machine_learning_task": "multiclass",
                         "neural_network_architecture": [4, 9, 12, 9, 3],
                         "hidden_layers_activation_function": "relu",
                         "optimizer": optimizer,
                         "metric": "Recall",
                         "batch_size": 10,
                         "epochs": 50}

# create artificial neural network configuration object
ann_handler_config = ArtificialNeuralNetworkHandlerConfig(neural_network_config)

# create artificial neural network handler object
ann_handler = ArtificialNeuralNetworkHandler(ann_handler_config)

# training designed neural network
ann_handler.train_neural_network(X_train, dummy_y_train)

# getting predictions
y_test_predictions = ann_handler.get_predictions(X_test, target_categories=[0, 1, 2])

# displaying classification report
print(classification_report(y_test, y_test_predictions))
```

### Randomized Search for Multi-class Classification Usage Example
```python
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
```

### Regression Usage Example
```python
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from artificial_neural_network_model_automation.artificial_neural_network_handler import ArtificialNeuralNetworkHandlerConfig
from artificial_neural_network_model_automation.artificial_neural_network_handler import ArtificialNeuralNetworkHandler
from helper.regression_helper import regression_report

# loading data
data = load_boston()
features = data.data
target = data.target

# splitting data
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.33, random_state=42)

# designing neural network

neural_network_config = {"machine_learning_task": "regression",
                         "neural_network_architecture": [13, 20, 20, 1],
                         "hidden_layers_activation_function": "relu",
                         "optimizer": "adam",
                         "metric": "mean_squared_error",
                         "batch_size": 10,
                         "epochs": 50}

# create artificial neural network configuration object
ann_handler_config = ArtificialNeuralNetworkHandlerConfig(neural_network_config)

# create artificial neural network handler object
ann_handler = ArtificialNeuralNetworkHandler(ann_handler_config)

# training designed neural network
ann_handler.train_neural_network(X_train, y_train)
# making predictions
y_pred = ann_handler.get_predictions(X_test)

number_of_features = X_train.shape[1]
regression_report(y_test, y_pred, number_of_features)
```

### Randomized Search for Regression Usage Example
```python
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
```

## Contributing
Suggestions for new features are welcome. For major changes or improvements, please open an issue first to 
discuss what you would like to change or improve.

## License
[MIT](https://github.com/AyberkYavuz/artificial_neural_network_automation/blob/main/LICENSE)

## In Memory of Mustafa Akgül
Mustafa Akgül (May 10, 1948 Ankara, Turkey - December 13, 2017 Ankara, Turkey) is a Turkish academic, engineer, 
mathematician, computer scientist, activist, known for his work for the spread of the Internet in Turkey.

He is known as the father of the Internet and free software in Turkey.

I have always been proud to be your student.
 
I dedicate my project to your unique memory.