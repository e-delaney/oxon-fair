"""Tests for FairPredictor"""

import numpy as np
import pandas as pd
import sklearn.linear_model
import anonfair as fair
from anonfair.utils import group_metrics as gm

classifier_type = sklearn.linear_model.LogisticRegression

train_data = pd.read_csv("https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv")
test_data = pd.read_csv("https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv")

# Merge and shuffle the data
total_data = pd.concat([train_data, test_data])
y = total_data["class"] == " >50K"
total_data = total_data.drop(columns="class")
total_data = pd.get_dummies(total_data)

train = total_data.sample(frac=0.5)
val_test = total_data.drop(train.index)
train_y = y.iloc[train.index]
val_test_y = y.drop(train_y.index)
val = val_test.sample(frac=0.4)
test = val_test.drop(val.index)
val_y = y.iloc[val.index]
test_y = val_test_y.drop(val.index)
predictor = classifier_type()
predictor.fit(train, train_y)

val_dict = {"data": val, "target": val_y}
test_dict = {"data": test, "target": test_y}

val_dict_g = fair.build_data_dict(val_y, val, val['sex_ Female'])
test_dict_g = fair.build_data_dict(test_y, test, test['sex_ Female'])


def test_slack_constraints(use_fast=True):
    """Slack constraints should not alter the solution found.
    In practice there seems to be some instability in the slow pathway and occasionally it does.
    Rerun and confirm there's a problem before debugging."""
    fpredictor = fair.FairPredictor(predictor, test_dict, "sex_ Female", use_fast=use_fast)
    cpredictor = fair.FairPredictor(predictor, test_dict, "sex_ Female", use_fast=use_fast)

    fpredictor.fit(gm.accuracy, gm.recall.diff, 0.005)
    cpredictor.fit(gm.accuracy, gm.recall.diff, 0.005, additional_constraints=((gm.pos_pred_rate, 0),))

    # Evaluate the change in fairness (recall difference corresponds to EO)
    measures = fpredictor.evaluate_fairness(verbose=False)
    cmeasures = cpredictor.evaluate_fairness(verbose=False)
    assert np.isclose(measures, cmeasures,).all().all()

    # check fit did something
    assert measures["original"]["recall.diff"] > 0.005
    assert measures["updated"]["recall.diff"] < 0.005


def test_slack_constraints_slow():
    test_slack_constraints(False)


# def test_slack_constraints_hybrid():
#    'Warning this consistency fails 50% of the time '
#    test_slack_constraints('hybrid')


def test_active_constraints(use_fast=True):
    """Active constraints should alter the solution found"""
    cpredictor = fair.FairPredictor(predictor, test_dict, "sex_ Female", use_fast=use_fast)

    cpredictor.fit(gm.accuracy, gm.recall.diff, 0.005, additional_constraints=((gm.pos_pred_rate, .7),))
    assert cpredictor.evaluate(metrics={'m': gm.pos_pred_rate})['updated'][0] > .7


def test_active_constraints_slow():
    test_active_constraints(False)


def test_active_constraints_hybrid():
    test_active_constraints('hybrid')
