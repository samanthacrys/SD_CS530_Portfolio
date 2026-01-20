import math
import random

def neuron(inputs, weights, bias, activation):
    """
    Implements a single artificial neuron with sigmoid or step activation.

    Args:
        inputs (list of float): The input values to the neuron.
        weights (list of float): The weights corresponding to each input.
        bias (float): The bias term for the neuron.
        activation (str): The activation function to use, either 'sigmoid' or 'step'.

    Returns:
        float: The output of the neuron after applying the activation function.
               For 'sigmoid', returns a value between 0 and 1.
               For 'step', returns 0 or 1.

    Raises:
        ValueError: If inputs/weights are not lists of numbers, lengths don't match,
                   or activation is invalid.
    """
    if not isinstance(inputs, list) or not all(isinstance(i, (int, float)) for i in inputs):
        raise ValueError("Inputs must be a list of numbers")
    if not isinstance(weights, list) or not all(isinstance(w, (int, float)) for w in weights):
        raise ValueError("Weights must be a list of numbers")
    if len(inputs) != len(weights):
        raise ValueError("Inputs and weights must have the same length")
    if not isinstance(bias, (int, float)):
        raise ValueError("Bias must be a number")
    z = sum(i * w for i, w in zip(inputs, weights)) + bias
    if activation == 'sigmoid':
        try:
            return 1 / (1 + math.exp(-z))
        except OverflowError:
            return 1.0 if z > 0 else 0.0
    elif activation == 'step':
        return 1 if z >= 0 else 0
    else:
        raise ValueError("Activation must be 'sigmoid' or 'step'")

def binary_classifier(dataset, weights, bias, activation):
    """
    Implements a binary classifier using the neuron function.

    Args:
        dataset (list of list): A list of samples, where each sample is [inputs, label].
                               inputs is a list of float, label is 0 or 1.
        weights (list of float): The weights for the neuron.
        bias (float): The bias for the neuron.
        activation (str): The activation function, 'sigmoid' or 'step'.

    Returns:
        list of float: The predictions for each sample in the dataset.

    Raises:
        ValueError: If dataset format is invalid or other inputs are incorrect.
    """
    if not isinstance(dataset, list):
        raise ValueError("Dataset must be a list")
    for sample in dataset:
        if not (isinstance(sample, list) and len(sample) == 2):
            raise ValueError("Each sample must be [inputs, label]")
        inputs, label = sample
        if not isinstance(inputs, list) or not all(isinstance(i, (int, float)) for i in inputs):
            raise ValueError("Inputs must be a list of numbers")
        if not isinstance(label, (int, float)) or label not in (0, 1):
            raise ValueError("Label must be 0 or 1")
    predictions = []
    for sample in dataset:
        inputs, label = sample
        pred = neuron(inputs, weights, bias, activation)
        predictions.append(pred)
    return predictions

def create_synthetic_dataset(n_samples=100):
    """
    Creates a synthetic dataset for binary classification.

    The dataset consists of 2D points (x1, x2) with labels based on x1 + x2 > 0.
    Uses a fixed random seed for reproducibility.

    Args:
        n_samples (int): The number of samples to generate. Must be positive.

    Returns:
        list of list: A list where each element is [inputs, label].
                     inputs is [x1, x2], label is 0 or 1.

    Raises:
        ValueError: If n_samples is not a positive integer.
    """
    if not isinstance(n_samples, int) or n_samples <= 0:
        raise ValueError("n_samples must be a positive integer")
    random.seed(42)  # For reproducibility
    dataset = []
    for _ in range(n_samples):
        x1 = random.uniform(-1, 1)
        x2 = random.uniform(-1, 1)
        label = 1 if x1 + x2 > 0 else 0
        dataset.append([[x1, x2], label])
    return dataset

def calculate_weights(dataset, activation, learning_rate=0.1, epochs=100):
    """
    Calculates the weights and bias for the neuron using training.

    For 'step' activation, uses the perceptron learning rule with early stopping if converged.
    For 'sigmoid' activation, uses simple gradient descent for logistic regression.

    Args:
        dataset (list of list): The training dataset, each sample [inputs, label].
        activation (str): The activation function, 'sigmoid' or 'step'.
        learning_rate (float): The learning rate for updates. Default 0.1.
        epochs (int): The number of training epochs. Default 100.

    Returns:
        tuple: (weights, bias) where weights is list of float, bias is float.

    Raises:
        ValueError: If dataset is invalid or activation is unsupported.
    """
    if not isinstance(dataset, list) or not dataset:
        raise ValueError("Dataset must be a non-empty list")
    n_features = len(dataset[0][0])
    for sample in dataset:
        if not (isinstance(sample, list) and len(sample) == 2):
            raise ValueError("Each sample must be [inputs, label]")
        inputs, label = sample
        if len(inputs) != n_features:
            raise ValueError("All samples must have the same number of features")
        if not isinstance(label, (int, float)) or label not in (0, 1):
            raise ValueError("Label must be 0 or 1")
    if not isinstance(learning_rate, (int, float)) or learning_rate <= 0:
        raise ValueError("Learning rate must be a positive number")
    if not isinstance(epochs, int) or epochs <= 0:
        raise ValueError("Epochs must be a positive integer")
    weights = [0.0] * n_features
    bias = 0.0
    for epoch in range(epochs):
        updated = False
        for inputs, label in dataset:
            if activation == 'step':
                z = sum(i * w for i, w in zip(inputs, weights)) + bias
                pred = 1 if z >= 0 else 0
                error = label - pred
                if error != 0:
                    weights = [w + learning_rate * error * i for w, i in zip(weights, inputs)]
                    bias += learning_rate * error
                    updated = True
            elif activation == 'sigmoid':
                pred = neuron(inputs, weights, bias, 'sigmoid')
                error = label - pred
                weights = [w + learning_rate * error * i for w, i in zip(weights, inputs)]
                bias += learning_rate * error
            else:
                raise ValueError("Activation must be 'sigmoid' or 'step'")
        if activation == 'step' and not updated:
            break  # Early stopping for perceptron
    return weights, bias

def calculate_accuracy(predictions, labels, threshold=0.5):
    """
    Calculates the accuracy of predictions against true labels.

    For sigmoid predictions, uses the threshold to convert to 0/1.

    Args:
        predictions (list of float): The predicted values.
        labels (list of int): The true labels (0 or 1).
        threshold (float): Threshold for sigmoid predictions. Default 0.5.

    Returns:
        float: The accuracy as a fraction.

    Raises:
        ValueError: If lengths don't match or types are invalid.
    """
    if not isinstance(predictions, list) or not isinstance(labels, list):
        raise ValueError("Predictions and labels must be lists")
    if len(predictions) != len(labels):
        raise ValueError("Predictions and labels must have the same length")
    if not all(isinstance(l, (int, float)) and l in (0, 1) for l in labels):
        raise ValueError("Labels must be 0 or 1")
    correct = 0
    for pred, label in zip(predictions, labels):
        if not isinstance(pred, (int, float)):
            raise ValueError("Predictions must be numbers")
        if isinstance(pred, int):
            pred_class = pred
        else:
            pred_class = 1 if pred >= threshold else 0
        if pred_class == label:
            correct += 1
    return correct / len(labels)

def train_test_split(dataset, test_size=0.2):
    """
    Splits the dataset into training and testing sets.

    Args:
        dataset (list of list): The full dataset, each sample [inputs, label].
        test_size (float): Proportion of the dataset to include in the test split (0 < test_size < 1).

    Returns:
        tuple: (train_set, test_set) where each is a list of [inputs, label].

    Raises:
        ValueError: If test_size is invalid or dataset is too small.
    """
    if not isinstance(dataset, list) or len(dataset) < 2:
        raise ValueError("Dataset must be a list with at least 2 samples")
    if not isinstance(test_size, (int, float)) or not (0 < test_size < 1):
        raise ValueError("test_size must be a float between 0 and 1")
    n_test = int(len(dataset) * test_size)
    if n_test == 0:
        raise ValueError("test_size too small for dataset size")
    # Simple split: first n_test for test, rest for train
    test_set = dataset[:n_test]
    train_set = dataset[n_test:]
    return train_set, test_set