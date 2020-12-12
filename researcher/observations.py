from collections import defaultdict

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

    def final_observations(self, key):
        """Identifies the values associated with the specified key and
        returns the last recorded value. If the key is associated with
        more than one fold, the last values of each fold are returned.

        Args:
            key (string): A name that is expected to be associated with
            some fold-related data collected from the experiment.

        Returns:
            object: The last recorded datapoint for the specified name for
            each fold.

        Raises:
            ValueError: If the specified key is not associated with a list
            of values.

            ValueError: If the specified key is associated with an empty 
            list.
        """

        values = self.observations[key]

        if not isinstance(values, list):
            raise ValueError(f"expected key {key} to be associated with a list of observations, got {values}")
        
        if len(values) == 0:
            raise ValueError(f"expected key {key} to have some values associated with it, got {values}")

        if isinstance(values[0], list):
            return [fold[-1] for fold in values]

        return values[-1]