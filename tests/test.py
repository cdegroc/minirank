import numpy as np
from scipy import stats
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import accuracy_score, auc_score, precision_score, recall_score
from minirank import sgd_train, sgd_predict

def demo():
    # Basic demo test
    X_train, y_train = load_svmlight_file("demo.train", query_id=False)
    X_test, y_test = load_svmlight_file("demo.train", query_id=False)
    coef, _ = sgd_train(X_train, y_train, np.ones(y_test.shape), alpha=0.1, n_features=150000, model='rank', max_iter=100000)
    preds = sgd_predict(X_test, coef, blocks=None)
    preds = np.sign(preds)
    assert accuracy_score(y_test, preds) > 0.98
    assert precision_score(y_test, preds) > 0.98
    assert recall_score(y_test, preds) > 0.98
    assert auc_score(y_test, preds) > 0.98

def test_1():
    np.random.seed(0)
    X = np.random.randn(200, 5)
    query_id = np.ones(len(X))
    w = np.random.randn(5)
    y = np.dot(X, w)
    coef, _ = sgd_train(X, y, query_id, 1., max_iter=100)
    prediction = sgd_predict(X, coef)
    tau, _ = stats.kendalltau(y, prediction)
    assert np.abs(1 - tau) > 1e-3

if __name__ == '__main__':
    test_1()
    demo()
