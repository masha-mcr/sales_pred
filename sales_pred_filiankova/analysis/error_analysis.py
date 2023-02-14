import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def target_prediction_scatter(target, pred):
    sns.set(rc={'figure.figsize': (10, 8)})
    error = np.abs(target - pred)
    sns.scatterplot(x=pred, y=target, hue=error)
    plt.xlabel('Prediction')
    plt.ylabel('Target')
    plt.title('Target vs. Prediction dependency')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    return plt.gcf()
