from pandas import read_csv
from sklearn.preprocessing import LabelEncoder
from artificial_neural_network_model_automation.classification_handler import ANNClassificationHandlerConfig
from artificial_neural_network_model_automation.classification_handler import ANNClassificationHandler
from sklearn.metrics import classification_report
from artificial_neural_network_model_automation.data_path_handler import get_data_path

data_name = "sonar.csv"
data_path = get_data_path(data_name, os_type="mac")


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
# designing neural network
ann_classification_handler_config = ANNClassificationHandlerConfig()
ann_classification_handler_config.classification_type = "binary"
ann_classification_handler_config.number_of_inputs = 60
ann_classification_handler_config.number_of_hidden_layers = 3
ann_classification_handler_config.dropout = False
ann_classification_handler_config.number_of_outputs = 1
ann_classification_handler_config.metric = "accuracy"
ann_classification_handler_config.batch_size = 10
ann_classification_handler_config.epochs = 50

ann_classification_handler = ANNClassificationHandler(ann_classification_handler_config)
# training neural network
ann_classification_handler.train_neural_network(X, encoded_Y)
# making predictions
y_pred = ann_classification_handler.classifier.predict(X)
# our threshold is 0.5. if number is bigger han 0.5, it returns true. If number is less than 0.5, it returns false
y_pred = [1 if prob > 0.5 else 0 for prob in y_pred]
# classification report
print(classification_report(encoded_Y, y_pred, target_names=encoder.classes_))
