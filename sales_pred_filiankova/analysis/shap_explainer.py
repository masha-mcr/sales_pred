import shap
import matplotlib.pyplot as plt


class ShapExplainer:
    def __init__(self, model):
        shap.initjs()
        self._explainer = shap.TreeExplainer(model)
        self._shap_values = None
        self._data = None

    def get_shap_values(self, data):
        self._data = data
        self._shap_values = self._explainer.shap_values(data)
        return self._shap_values

    def summary_plot(self, max_display=20):
        plt.figure()
        shap.summary_plot(self._shap_values, self._data, feature_names=self._data.columns, max_display=max_display,
                          show=False)
        return plt.gcf()

    def plot_prediction(self, data, feature_names):
        plt.figure()
        error_shap_value = self._explainer.shap_values(data.values.reshape(1, -1))
        plot = shap.force_plot(self._explainer.expected_value, error_shap_value, data.values,
                               feature_names=feature_names, show=False,
                               matplotlib=True)
        return plot
