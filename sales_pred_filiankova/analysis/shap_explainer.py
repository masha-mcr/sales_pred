import shap


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
        shap.summary_plot(self._shap_values, self._data, feature_names=self._data.columns,  max_display=max_display)

    def plot_prediction(self, data):
        error_shap_value = self._explainer.shap_values(data)
        shap.force_plot(self._explainer.expected_value, error_shap_value, data.values, feature_names=data.columns)


