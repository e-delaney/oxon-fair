# Examples
In this folder we provide an overview of the functionality of our toolkit in several notebooks. The file and corresponding descriptions are as follows:

| File Name                                         | Description                                                                                                                                                                                                 |
|---------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `training_nlp_jigsaw_religion.ipynb`              | This code demonstrates how to enforce fairness when working with deep neural networks in NLP. It shows an example on Jigsaw with the attribute 'Religion' as the protected group and targets hate speech presence. |
| `adult_fairlearn_comparision.ipynb`               | This notebook compares the overfitting of Fairlearn Vs AnonFair using random forests and decision trees on the adult dataset, with sex as the protected attribute.                                             |
| `quickstart_autogluon.ipynb`                      | This file contains demo code for FairPredictor with Autogluon, handling more fairness over multiple groups and enforcing a range of fairness definitions on COMPAS. It supports a wide range of performance metrics and fairness criteria without requiring access to protected attributes at test time. |
| `high-dim_fairlearn_comparision.ipynb`            | This notebook compares the overfitting of Fairlearn Vs AnonFair on a resampled version of the myocardial infarction dataset, using sex as the protected attribute to induce unfairness.                         |
| `levelling_up.ipynb`                              | This code implements a new form of fairness that "levels up" instead of equalizing harms across groups, reducing harm to individuals. It compares conventional fairness algorithms with new approaches that decrease harm. |
| `quickstart_xgboost.ipynb`                        | This file contains demo code for FairPredictor with XGBoost, handling more fairness over multiple groups and enforcing a range of fairness definitions on COMPAS. It supports a wide range of performance metrics and fairness criteria without requiring access to protected attributes at test time. |
| `quickstart_DeepFairPredictor_computer_vision.ipynb` | This code demonstrates how to enforce fairness when working with deep neural networks in computer vision, using datasets like CelebA and Fitzpatrick-17k. It allows specifying target attributes and metrics to measure fairness and performance. |
| `multi_group_fairlearn_comparision.ipynb`         | This notebook imports various libraries and datasets, training an XGBoost classifier on the adult dataset with 'race' as the protected attribute to evaluate fairness.                                        |


## Install
To run the example notebooks, you should first install the dependency packages with the following command:
```
pip install scikit-learn matplotlib xgboost fairlearn autogluon ucimlrepo
```