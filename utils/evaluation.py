from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from scipy.stats import ks_2samp

def evaluate_classifier(train_X, train_y, test_X, test_y):
    clf = LogisticRegression(max_iter=500)
    clf.fit(train_X, train_y)
    preds = clf.predict(test_X)
    return accuracy_score(test_y, preds)

def ks_test_distribution(real_data, synthetic_data):
    results = []
    for i in range(real_data.shape[1]):
        statistic, p_value = ks_2samp(real_data[:, i], synthetic_data[:, i])
        results.append((i, statistic, p_value))
    return results
