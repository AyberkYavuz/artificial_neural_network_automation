from helper.helper import contol_instance_type
from helper.helper import check_machine_learning_value
from helper.classification_helper import classification_scoring_dictionary
from helper.regression_helper import regression_scoring_dictionary
from helper.classification_helper import check_target_categories
from helper.helper import is_list_empty
from helper.helper import is_number_positive
from helper.helper import check_n_jobs
from helper.helper import check_instance_type_of_scoring
from helper.helper import check_value_of_scoring
from helper.helper import get_value_of_scoring_none_condition
from helper.decorators import execution_time
from artificial_neural_network_model_automation.artificial_neural_network_handler import ArtificialNeuralNetworkHandlerConfig
from artificial_neural_network_model_automation.artificial_neural_network_handler import ArtificialNeuralNetworkHandler
from random import choice
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
import pandas as pd
from keras.utils import np_utils


class ANNRandomizedSearchConfig:
    """A configuration for ANNRandomizedSearch instance.

    Attributes:
      machine_learning_task: str. The type of machine learning task. It takes 3 different values
                         which are "binary", "multiclass", "regression".
      scoring: str or None. It can be "accuracy", "roc_auc", "f1", "precision", "recall" or None. If None,
            its value becomes "f1". It is the selection criteria for best model.
      neural_network_architecture_list: List of neural network architectures that are represented by a python list. For example;
                                [[60, 70, 80, 1], [60, 80, 1], [60, 70, 70, 1]]
      hidden_layers_activation_function_list: List of hidden layers activation function types. For example;
                                          ["sigmoid", "relu", "tanh"]
      dropout_rate_list: List of dropout_rate values. For example; [None, 0.01, 0.0001, 0.1]
      optimizer_list: List of optimizers. For example; ["adam", "sgd", "nadam"] or
                  [keras.optimizers.Adam(learning_rate=0.01), keras.optimizers.Adam(learning_rate=0.02)]
      metric_list: List of metrics. For example; ["accuracy", "Recall", "Precision", "AUC"] or
                [tf.keras.metrics.AUC(), tf.keras.metrics.Precision()]
      batch_size_list: List of batch sizes. For example; [32, 64, 128]
      epochs_list: List of epochs. For example; [10, 20, 30, 40, 50, 100]
    """
    def __init__(self, neural_network_config_list_dict: dict):
        """Constructs all the necessary attributes for the ann_randomized_search_config object.

        Args:
          neural_network_config_list_dict: Python dictionary which includes neural network configuration lists data
        """
        self.machine_learning_task = neural_network_config_list_dict["machine_learning_task"]
        scoring = neural_network_config_list_dict.get("scoring")
        self.scoring = scoring
        self.neural_network_architecture_list = neural_network_config_list_dict["neural_network_architecture_list"]
        self.hidden_layers_activation_function_list = neural_network_config_list_dict["hidden_layers_activation_function_list"]
        self.dropout_rate_list = neural_network_config_list_dict["dropout_rate_list"]
        self.optimizer_list = neural_network_config_list_dict["optimizer_list"]
        self.metric_list = neural_network_config_list_dict["metric_list"]
        self.batch_size_list = neural_network_config_list_dict["batch_size_list"]
        self.epochs_list = neural_network_config_list_dict["epochs_list"]

    @property
    def machine_learning_task(self):
        return self._machine_learning_task

    @machine_learning_task.setter
    def machine_learning_task(self, ml_task):
        contol_instance_type(ml_task, "machine_learning_task", str)
        check_machine_learning_value(ml_task)
        self._machine_learning_task = ml_task

    @property
    def scoring(self):
        return self._scoring

    @scoring.setter
    def scoring(self, sc):
        check_instance_type_of_scoring(sc)
        check_value_of_scoring(sc, self.machine_learning_task)
        sc = get_value_of_scoring_none_condition(sc, self.machine_learning_task)
        self._scoring = sc

    @property
    def neural_network_architecture_list(self):
        return self._neural_network_architecture_list

    @neural_network_architecture_list.setter
    def neural_network_architecture_list(self, nna_list):
        variable_name = "neural_network_architecture_list"
        contol_instance_type(nna_list, variable_name, list)
        is_list_empty(nna_list, variable_name)
        self._neural_network_architecture_list = nna_list

    @property
    def hidden_layers_activation_function_list(self):
        return self._hidden_layers_activation_function_list

    @hidden_layers_activation_function_list.setter
    def hidden_layers_activation_function_list(self, hlaf_list):
        variable_name = "hidden_layers_activation_function_list"
        contol_instance_type(hlaf_list, variable_name, list)
        is_list_empty(hlaf_list, variable_name)
        self._hidden_layers_activation_function_list = hlaf_list

    @property
    def dropout_rate_list(self):
        return self._dropout_rate_list

    @dropout_rate_list.setter
    def dropout_rate_list(self, d_list):
        variable_name = "dropout_rate_list"
        contol_instance_type(d_list, variable_name, list)
        is_list_empty(d_list, variable_name)
        self._dropout_rate_list = d_list

    @property
    def optimizer_list(self):
        return self._optimizer_list

    @optimizer_list.setter
    def optimizer_list(self, o_list):
        variable_name = "optimizer_list"
        contol_instance_type(o_list, variable_name, list)
        is_list_empty(o_list, variable_name)
        self._optimizer_list = o_list

    @property
    def metric_list(self):
        return self._metric_list

    @metric_list.setter
    def metric_list(self, m_list):
        variable_name = "metric_list"
        contol_instance_type(m_list, variable_name, list)
        is_list_empty(m_list, variable_name)
        self._metric_list = m_list

    @property
    def batch_size_list(self):
        return self._batch_size_list

    @batch_size_list.setter
    def batch_size_list(self, bs_list):
        variable_name = "batch_size_list"
        contol_instance_type(bs_list, variable_name, list)
        is_list_empty(bs_list, variable_name)
        self._batch_size_list = bs_list

    @property
    def epochs_list(self):
        return self._epochs_list

    @epochs_list.setter
    def epochs_list(self, e_list):
        variable_name = "epochs_list"
        contol_instance_type(e_list, variable_name, list)
        is_list_empty(e_list, variable_name)
        self._epochs_list = e_list


class ANNRandomizedSearch:
    """Randomized Search which was designed for ANNHandler.

    Attributes:
      ann_randomized_search_config: ANNClassificationRandomizedSearchConfig instance.
      n_iter: int, default=10. Number of parameter settings that are sampled. n_iter trades off runtime vs quality of
                        the solution.
      n_jobs: int, default=None. The maximum number of concurrently running jobs, such as the number of Python worker
                        processes when backend=”multiprocessing” or the size of the thread-pool when
                        backend=”threading”. If -1 all CPUs are used. If 1 is given, no parallel computing code is used
                        at all, which is useful for debugging. For n_jobs below -1, (n_cpus + 1 + n_jobs) are used.
                        Thus for n_jobs = -2, all CPUs but one are used. None is a marker for ‘unset’ that will be
                        interpreted as n_jobs=1 (sequential execution) unless the call is performed under a
                        parallel_backend context manager that sets another value for n_jobs.
      best_param_: ANNClassificationHandlerConfig instance, default=None. This attribute will be initialized,
                        when _set_metric_params private method is called.
      best_estimator_: Keras neural network, default=None. This attribute will be initialized,
                        when _set_metric_params private method is called.
      best_score_: float, default=None. This attribute will be initialized,
                        when _set_metric_params private method is called.
    """
    def __init__(self, ann_randomized_search_config: ANNRandomizedSearchConfig,
                 n_iter=10, n_jobs=None):
        """Constructs all the necessary attributes for the ann_randomized_search object.

        Args:
          ann_randomized_search_config: ANNClassificationRandomizedSearchConfig instance.
          n_iter: int, default=10.
          n_jobs: int, default=None.
        """
        self.ann_randomized_search_config = ann_randomized_search_config
        self.n_iter = n_iter
        self.n_jobs = n_jobs
        self.best_param_ = None
        self.best_estimator_ = None
        self.best_score_ = None

    @property
    def n_iter(self):
        return self._n_iter

    @n_iter.setter
    def n_iter(self, n):
        variable_name = "n_iter"
        contol_instance_type(n, variable_name, int)
        is_number_positive(n, variable_name)
        self._n_iter = n

    @property
    def n_jobs(self):
        return self._n_jobs

    @n_jobs.setter
    def n_jobs(self, jobs):
        check_n_jobs(jobs)
        self._n_jobs = jobs

    def _get_randomly_ann_classification_handler_config(self):
        """Randomly creates ann_classification_handler_config based on atrributes.
        Returns:
            ann_handler_config: ArtificialNeuralNetworkHandlerConfig instance.
        """
        neural_network_architecture = choice(self.ann_randomized_search_config.neural_network_architecture_list)
        hidden_layers_activation_function = choice(self.ann_randomized_search_config.hidden_layers_activation_function_list)
        dropout_rate = choice(self.ann_randomized_search_config.dropout_rate_list)
        optimizer = choice(self.ann_randomized_search_config.optimizer_list)
        metric = choice(self.ann_randomized_search_config.metric_list)
        batch_size = choice(self.ann_randomized_search_config.batch_size_list)
        epochs = choice(self.ann_randomized_search_config.epochs_list)
        neural_network_config = {"machine_learning_task": self.ann_randomized_search_config.machine_learning_task,
                                 "neural_network_architecture": neural_network_architecture,
                                 "hidden_layers_activation_function": hidden_layers_activation_function,
                                 "dropout_rate": dropout_rate,
                                 "optimizer": optimizer,
                                 "metric": metric,
                                 "batch_size": batch_size,
                                 "epochs": epochs}
        ann_handler_config = ArtificialNeuralNetworkHandlerConfig(neural_network_config)
        return ann_handler_config

    def _fit_and_score(self, X_train, X_test, y_train, y_test, target_categories=None):
        """Fits and scores Keras neural network.
        Args:
            X_train: it can be list, numpy array, scipy-sparse matrix or pandas dataframe.
            X_test: it can be list, numpy array, scipy-sparse matrix or pandas dataframe.
            y_train: it can be list, numpy array, scipy-sparse matrix or pandas dataframe.
            y_test: it can be list, numpy array, scipy-sparse matrix or pandas dataframe.
            target_categories: list. Categories of target variable. If the task is multi-class classification,
                               this argument must be initialized.
        Returns:
            result: dict. It contains ANNClassificationHandlerConfig instance, float score and Keras neural network.
        """
        ann_handler_config = self._get_randomly_ann_classification_handler_config()
        ann_handler = ArtificialNeuralNetworkHandler(ann_handler_config)
        score = None
        if self.ann_randomized_search_config.machine_learning_task == "binary":
            ann_handler.train_neural_network(X_train, y_train)
            y_pred = ann_handler.get_predictions(X_test)
            scoring_method = classification_scoring_dictionary[self.ann_randomized_search_config.scoring]
            score = scoring_method(y_test, y_pred)
        elif self.ann_randomized_search_config.machine_learning_task == "multiclass":
            # dummy target transformation
            dummy_y_train = np_utils.to_categorical(y_train)
            ann_handler.train_neural_network(X_train, dummy_y_train)
            y_pred = ann_handler.get_predictions(X_test, target_categories=target_categories)
            scoring_method = classification_scoring_dictionary[self.ann_randomized_search_config.scoring]
            score = scoring_method(y_test, y_pred, average="macro")
        else:
            ann_handler.train_neural_network(X_train, y_train)
            y_pred = ann_handler.get_predictions(X_test)
            scoring_method = regression_scoring_dictionary[self.ann_randomized_search_config.scoring]
            if self.ann_randomized_search_config.scoring == "adjusted_r2":
                number_of_features = X_train.shape[1]
                score = scoring_method(y_test, y_pred, number_of_features)
            else:
                score = scoring_method(y_test, y_pred)

        result = {"score": score,
                  "ann_handler_config": ann_handler_config,
                  "model": ann_handler.neural_network}
        return result

    def _set_metric_params(self, list_of_dictionaries):
        """Initializes best_estimator_, best_param_ and best_score_ attributes.
        Args:
            list_of_dictionaries: list. It is list of dictionaries which contains score, ann_handler_config
                                  and model attributes.
        """
        metric_dataframe = pd.DataFrame(list_of_dictionaries)
        max_score = metric_dataframe["score"].max()
        data = metric_dataframe[metric_dataframe["score"] == max_score].iloc[0]
        self.best_estimator_ = data["model"]
        self.best_param_ = data["ann_handler_config"]
        self.best_score_ = max_score

    @execution_time("minutes")
    def fit(self, X, y, target_categories=None):
        """Run fit with all sets of parameters.

        Args:
        X : array-like of shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like of shape (n_samples, n_output) \
            or (n_samples,), default=None
            Target relative to X for classification or regression;
            None for unsupervised learning.
        target_categories : list. Categories of target variable. If the task is multi-class classification,
                            this argument must be initialized.
        Raises:
            Exception: if machine_learning_task condition is not met.
        """
        if self.ann_randomized_search_config.machine_learning_task == "multiclass":
            check_target_categories(target_categories)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
        list_of_dictionaries = list()
        if self.n_jobs is None:
            for _ in range(0, self.n_iter):
                result = self._fit_and_score(X_train, X_test, y_train, y_test,
                                             target_categories=target_categories)
                list_of_dictionaries.append(result)
                self._set_metric_params(list_of_dictionaries)
        else:
            list_of_dictionaries = Parallel(n_jobs=self.n_jobs)(
                delayed(self._fit_and_score)(
                    X_train, X_test, y_train, y_test, target_categories=target_categories
                )
                for _ in range(0, self.n_iter)
            )
            self._set_metric_params(list_of_dictionaries)

