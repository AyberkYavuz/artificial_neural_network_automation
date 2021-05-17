import pandas as pd
from numpy import argmax


def get_label_based_on_thresold(x, thresold):
   result = None
   if x > thresold:
      result = 1
   else:
      result = 0
   return result


def get_predictions_from_dummy_prob_matrix(dummy_prob_matrix, prediction_column_names, thresold=0.5):
   dummy_prob_matrix_df = pd.DataFrame(dummy_prob_matrix, columns=prediction_column_names)
   for column in prediction_column_names:
      dummy_prob_matrix_df[column] = dummy_prob_matrix_df[column].apply(lambda x: get_label_based_on_thresold(x, thresold=thresold))

   dummy_y_train_pred = dummy_prob_matrix_df.to_numpy()
   predictions = argmax(dummy_y_train_pred, axis=1)
   return predictions