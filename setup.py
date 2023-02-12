from setuptools import find_packages, setup

setup(
        name="sales_pred_filiankova",
        version="0.1.0",
        packages=find_packages(),
        description="test package upload",
        url="https://github.com/wangonya/contacts-cli",
        author="Mariya Filiankova",
        author_email="maryia.filiankova@gmail.com",
        license="MIT",
        classifiers=[
            "License :: OSI Approved :: MIT License",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.9",
        ],
        install_requires=[
            "pandas",
            "numpy",
            "shap",
            "xgboost",
            "scikit-learn",
            "hyperopt",
            "matplotlib",
            "seaborn",
            "boruta",
            "category_encoders",
            "tqdm"
            ]
        )
