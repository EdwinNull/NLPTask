import string

import numpy as np
import pandas as pd
import random
from tqdm import tqdm


# logistic regression class
class LogisticRegression:
    def __init__(self, learning_rate=0.05, num_iter=100):
        self.learning_rate = learning_rate
        self.num_iter = num_iter
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        m, n = X.shape
        num_classes = len(np.unique(y))
        self.weights = np.zeros((n, num_classes))
        self.bias = np.zeros(num_classes)
        y_onehot = np.eye(num_classes, dtype=int)[y.astype(int)]

        for i in range(self.num_iter):
            z = np.dot(X, self.weights) + self.bias
            y_pred = softmax(z)
            loss = cross_entropy_loss(y_onehot, y_pred)
            dw, db = calculate_grad(X, y_onehot, y_pred)
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            if i % 10 == 0:
                print(f'Iteration: {i}, Loss: {loss:.4f}')

    def predict(self, X):
        z = np.dot(X, self.weights) + self.bias
        y_pred = softmax(z)
        return np.argmax(y_pred, axis=1)


# use N-gram method to extract features
# build ngram vocabulary
def build_ngram_vocab(data, n):
    ngram_vocab = {}
    index = 0
    for text, label in data:
        # convert text to lower case and remove punctuation
        text = text.lower().translate(str.maketrans('', '', string.punctuation))
        tokens = text.split()

        # go through vocabulary and build ngram
        for i in range(len(tokens) - n + 1):
            # build ngram tuple
            ngram = tuple(tokens[i:i + n])
            if ngram not in ngram_vocab:
                # if ngram not in vocabulary, add it to vocabulary and assign an index
                ngram_vocab[ngram] = index
                index += 1
    return ngram_vocab


def text_to_ngram_vector(text, ngram_vocab, n):
    vector = np.zeros(len(ngram_vocab))
    tokens = text.lower().split()
    for i in range(len(tokens) - n + 1):
        ngram = tuple(tokens[i:i + n])
        if ngram in ngram_vocab:
            vector[ngram_vocab[ngram]] += 1
    return vector


# data process
def data_process(file_path, n, sample_rate=0.1):
    df = pd.read_csv(file_path, sep='\t')
    # sample data
    df = df.sample(frac=sample_rate, random_state=42)
    # transform in appropriate form
    data = [(row['Phrase'], row['Sentiment']) for _, row in df.iterrows()]
    ngram_vocab = build_ngram_vocab(data, n)
    process_bar = tqdm(total=len(data), desc='Processing data', unit=" samples")
    # transform text to ngram vector
    X = np.empty((len(data), len(ngram_vocab)), dtype=np.int16)
    y = np.empty((len(data)), dtype=np.int16)
    for i, (text, label) in enumerate(data):
        ngram_vector = text_to_ngram_vector(text, ngram_vocab, n)
        X[i] = ngram_vector
        y[i] = label
        process_bar.update(1)
    process_bar.close()
    return X, y, ngram_vocab


# split data into train and test set
def train_test_split(X, y, test_size=0.2, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    # get sample number of test set
    num_samples = X.shape[0]
    num_test_samples = int(num_samples * test_size)
    # random generate index
    indices = np.random.permutation(num_samples)

    test_indices = indices[:num_test_samples]
    train_indices = indices[num_test_samples:]
    X_train, y_train = X[train_indices], y[train_indices]
    X_test, y_test = X[test_indices], y[test_indices]

    return X_train, y_train, X_test, y_test


# softmax function
def softmax(array):
    exp_array = np.exp(array - np.max(array, axis=1, keepdims=True))
    return exp_array / np.sum(exp_array, axis=1, keepdims=True)


# cross-entropy loss function
def cross_entropy_loss(y_true, y_pred):
    epsilon = 1e-7
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    loss = -np.sum(np.sum(y_true * np.log(y_pred), axis=1))
    return loss


# grad calculate function
def calculate_grad(X, y_true, y_pred):
    m = y_true.shape[0]
    grad = np.dot(X.T, (y_pred - y_true)) / m
    return grad, np.mean(y_pred - y_true, axis=0)
    # return grad and loss


def train():
    file_path = "/home/edwin/Downloads/train.tsv"
    n = 2
    X, y, _ = data_process(file_path, n)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(learning_rate=0.05, num_iter=100)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    accuracy = np.mean(y_pred == y_val)
    print(f'Accuracy: {accuracy:.4f}')


if __name__ == '__main__':
    train()