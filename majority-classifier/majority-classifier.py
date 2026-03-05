import numpy as np

def majority_classifier(y_train, X_test):
    """
    Predict the most frequent label in training data for all test samples.
    """
    # Write code here
    y_train = np.asanyarray(y_train)
    if y_train.size == 0:
        return np.array([])
    uniques, counts = np.unique(y_train, return_counts=True)
    majority_label = uniques[np.argmax(counts)]
    num_test_samples = len(X_test)
    predictions = np.full(num_test_samples, majority_label, dtype=y_train.dtype)
    return predictions