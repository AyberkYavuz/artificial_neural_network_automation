from pandas import read_csv
from sklearn.preprocessing import LabelEncoder
from artificial_neural_network_model_automation.classification_handler import ANNClassificationHandler
from sklearn.metrics import classification_report
from artificial_neural_network_model_automation.data_path_handler import get_data_path

data_name = "sonar.csv"
data_path = get_data_path(data_name)


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
ann_classification_handler = ANNClassificationHandler("binary", 60, 3, False, 1, "accuracy")
classifier = ann_classification_handler.design_neural_network()
# training neural network
classifier.fit(X, encoded_Y, batch_size=10, epochs=50)
# making predictions
y_pred = classifier.predict(X)
# our threshold is 0.5. if number is bigger han 0.5, it returns true. If number is less than 0.5, it returns false
y_pred = [1 if prob > 0.5 else 0 for prob in y_pred]
# classification report
print(classification_report(encoded_Y, y_pred, target_names=encoder.classes_))
