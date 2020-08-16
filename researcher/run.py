import datetime
import os
import time
import gc
import json
import copy

import numpy as np

from researcher.fileutils import *
from researcher.globals import *

def reduced_params(params, unwanted_keys):
    """Create a copy of the given parameters without descriptive (human-directed) fields.
    """

    if not isinstance(unwanted_keys, set):
        unwanted_keys = set(unwanted_keys)

    return {k: params[k] for k in params.keys() - unwanted_keys}

def run_experiment(params, experiment_fn, save_path):
    cloned_params = copy.deepcopy(params)
    param_hash = get_hash(cloned_params)

    print("Commencing experiment {}\n".format(param_hash))

    # experiment_fn must return a Results instance.
    results = experiment_fn(reduced_params(cloned_params, ['title', 'notes']), identifier=param_hash)

    cloned_params["hash"] = param_hash
    cloned_params["timestamp"] = datetime.datetime.now().strftime(DATE_FORMAT)

    if "title" in cloned_params:
        title = cloned_params["title"]
    else:
        title = "no_title"

    save_experiment(save_path, "{}_{}".format(title, param_hash), parameters=cloned_params, results=results.view())
    
    print("Finished experiment {}\n".format(param_hash))

    return results

