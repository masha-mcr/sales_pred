import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def target_prediction_scatter(target, pred):
    error_data = pd.DataFrame({'target': target, 'pred': pred})
    error_data['error'] = np.abs(target - pred)
    plt.figure()
    sns.scatterplot(data=error_data, x='pred', y='target', hue='error')
    plt.xlabel('Prediction')
    plt.ylabel('Target')
    plt.title('Target vs. Prediction dependency')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    return plt.gcf()


def get_biggest_errors(val_x, val_y, val_pred, frac=0.01):
    val_with_errors = val_x.copy()
    val_with_errors['target'] = val_y
    val_with_errors['pred'] = val_pred
    val_with_errors['error'] = val_with_errors['target'] - val_with_errors['pred']
    val_with_errors['error^2'] = val_with_errors['error']**2
    val_with_errors.sort_values(by=['error^2'], inplace=True, ascending=False)
    return val_with_errors[:int(val_with_errors.shape[0]*frac)]


def get_problematic_categories(val_with_errors, val_orig, cols_list, k=5):
    val_orig = val_orig.loc[val_with_errors.index, :]
    problematic = {}
    for column in cols_list:
        problematic[column] = list(val_orig[column].value_counts().index[:k])

    return problematic


def plot_errors_across_target(val_with_errors, bins=[0, 1, 2, 10, 50, 100]):
    val_with_errors['target_binned'] = pd.cut(val_with_errors['target'], bins=bins)
    plt.figure()
    plt.title('Errors distribution across targets')
    sns.set(rc={'figure.figsize': (16, 11)})
    sns.violinplot(data=val_with_errors, x='target_binned', y='error')
    plt.axhline(y=0.0, color='r')
    return plt.gcf()

