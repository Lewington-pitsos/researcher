from collections import defaultdict

import numpy as np

from researcher.globals import *

class Observations():
    """A light wrapper around experiment observations.

    Attributes:
        observations (dict, optional): All observations made during the experiment.

    """
    def __init__(self, observations=None):
        self.observations = observations if observations else {}

class ObservationBuilder(Observations):
    """Makes the task of gathering and saving experiment results easier.
    """
    def __init__(self):
        super().__init__(None)

        self.__fold_data = set()
        self.__non_fold_data = set()

    def __add_fold_value(self, fold, name, value):
        self.observations[name][fold].append(value)

    def __add_fold(self, fold, name):
        if not name in self.observations:
            self.observations[name] = []
            self.__fold_data.add(name)
        
        while len(self.observations[name]) <= fold:
            self.observations[name].append([])

    def set_observation(self, key, value):
        """Stores the specified values under the specified key. 

        Args:
            name (string): The type of data being logged, e.g.: "f1_score"
            or  "loss". 

            value (object): The observations to store under the given key.

        Raises:
            ValueError: If the given key is already being used to store values.
        """
        if key in self.observations:
            raise ValueError(f"key {key} is already being used to store the following observations: {self.observations[key]}")

        self.__non_fold_data.add(key)
        self.observations[key] = value

    def add_fold_observation(self, fold, key, value):
        """Appends the given value to the list of values associated with 
        the specified field in the specified fold. For instance, you might
        call this method to continuously add datapoints to the accuracy 
        metric for each fold as training progresses. You cannot add a name
        to the fold results if it already exists in general results or
        vise versa.

        Args:
            fold (int): The fold to add data to. The first fold is fold 0.

            name (string): The name of the data being logged, e.g.:
            "f1_score" or  "loss". 

            value (object): The next value to add to the data for this 
            fold. Usually a float.
        Raises:
            ValueError: If the specified name is already storing non-fold
            related observations.

        """
        if key in self.__non_fold_data:
            raise ValueError(f"Cannot add fold data to {key}, since this key is already being used to store non-fold related observations")

        self.__add_fold(fold, key)
        self.__add_fold_value(fold, key, value)

    def add_multiple(self, fold, name, values):
        """Appends multiple values to the list of values associated with 
        the specified field in the specified fold.

        Args:
            fold (int): The fold to add data to (usually the current 
            fold).

            name (string): The name of the data being logged, e.g.:
            "f1_score" or  "loss".

            values (list[object]): The next values to add to the data for this 
            fold. Usually a list of floats.
        """
        self.__add_fold(fold, name)

        for value in values:
            self.__add_fold_value(fold, name, value)

    def active_fold(self):
        """
        Returns:
            int: The index of the highest fold added to so far.
        """
        return len(self.fold_results) - 1

class FinalizedObservations(Observations):
    """Observations loaded from an already completed experiment. 
    """
    def __init__(self, observations):
        super().__init__(observations)

    def get_observations(self, name):
        """Returns the values associated with the specified name.

        Args:
            name (string): The name of the data to be returned. Often a 
            metric like "loss" or "accuracy".

        Returns:
            object: The data associated with the specified name.
        """
        return self.observations[name]

    def has_observation(self, name):
        """
        Args:
            name (string): A name that is expected to be associated with
            some data collected from the experiment.

        Returns:
            bool: An indicator of whether there is any data associated with the 
            specified name stored in the experiment results.
        """
        return name in self.observations

    def get_final_metric_values(self, name):
        """Identifies the values in each fold associated with the 
        specified name and returns the last recorded value for each fold.

        Args:
            name (string): A name that is expected to be associated with
            some fold-related data collected from the experiment.

        Returns:
            object: The last recorded datapoint for the specified name for
            each fold.
        """
        return [metrics[name][-1] for metrics in self.fold_results]

    def get_fold_aggregated_metric(self, name, agg_fn):
        """Aggregates the data associated with the specified name over all
        folds using the specified function and returns the aggregation.

        Args:
            name (string): A name that is expected to be associated with
            some fold-related data collected from the experiment.

            agg_fn (Callable): A function that can be used to aggregate 
            numpy arrays.

        Returns:
            numpy.ndarray: The data associated with the specified name 
            aggregated accross all folds.
        """
        fold_wise = []
        for metrics in self.fold_results:
            fold_wise.append(metrics[name])

        return agg_fn(np.array(fold_wise), axis=0)
    

