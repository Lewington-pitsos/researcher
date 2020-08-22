import matplotlib.pyplot as plt
from scipy import stats

from researcher.fileutils import *

def display_results(record_path, hash_segment, metric):
    e = past_experiment_from_hash(record_path, hash_segment)
    fig, ax = plt.subplots()

    print(np.mean(e.get_metric(metric)))

    for fold in e.get_metric(metric):
        ax.scatter(0, np.array(fold))
    
    return e

def scatter_compare(experiments, metrics, **kwargs):
    fig, axes = plt.subplots(len(metrics), **kwargs)
    
    for i, metric in enumerate(metrics):  
        print("\n" + metric)
        for e in experiments:
            if e.has_metric(metric):
                scores = e.get_final_metric_values(metric)
                labels = [f"fold_{i}" for i in range(len(scores))]
                scores += [np.mean(scores)]
                labels += ["mean"]
                axes[i].plot(labels, scores[:])
                print( "mean", scores[-1], e.identifier())
        axes[i].grid()
            
    fig.legend([e.identifier() for e in experiments])

def plot_lr(e, metric):
    _, ax = plt.subplots(figsize=(20, 20))
    
    ax.plot(e.get_metric(metric)[0])
    ax.plot(e.get_metric('learning_rate')[0])
    
    best_index = np.argmin(e.get_metric(metric)[0])
    print("lowerst loss achieved at: ", best_index)
    print("corresponding lr: ", e.get_metric(metric)[0][best_index])


def plot_training(e, metric, **kwargs):
    trn = e.get_metric(metric)
    val = e.get_val_metric(metric)

    _, ax = plt.subplots(len(trn) + 1, **kwargs)
    
    for i in range(len(trn)):
        ax[i].plot(trn[i])
        ax[i].plot(val[i])
    
    ax[len(trn)].plot(np.mean(np.array(trn), axis=0))
    ax[len(trn)].plot(np.mean(np.array(val), axis=0))

def compare_training(es, metric, **kwargs):
    fig, ax = plt.subplots(**kwargs)
    labels = []
    
    for e in es:
        print(e.identifier())
        print(metric, np.mean(e.get_final_metric_value(metric)))
        print("val_" + metric, np.mean(e.get_final_val_metric_value(metric)))
        
        
        trn = e.get_metric(metric)
        val = e.get_val_metric(metric)
        ax.plot(np.mean(np.array(trn), axis=0))
        ax.plot(np.mean(np.array(val), axis=0))
        labels += [e.identifier() + "_" + metric, e.identifier() + "_val_" + metric, ]

    ax.legend(labels)
