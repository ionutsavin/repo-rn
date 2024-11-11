import numpy as np
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt


def normalize_data(data):
    return np.array(data) / 255.0


def one_hot_encode(labels, num_classes=10):
    one_hot_labels = np.zeros((len(labels), num_classes))
    one_hot_labels[np.arange(len(labels)), labels] = 1
    return one_hot_labels


def load_mnist(is_train: bool):
    dataset = MNIST(root='data/', transform=lambda x: np.array(x).flatten(), train=is_train, download=True)
    mnist_data, mnist_labels = [], []
    for image, label in dataset:
        mnist_data.append(image)
        mnist_labels.append(label)
    mnist_data = normalize_data(mnist_data)
    mnist_labels = one_hot_encode(mnist_labels)
    return np.array(mnist_data), np.array(mnist_labels)


def load_mnist_with_validation(validation_ratio=0.2):
    train_data, train_labels = load_mnist(is_train=True)
    num_validation = int(len(train_data) * validation_ratio)
    indices = np.random.permutation(len(train_data))
    val_indices, train_indices = indices[:num_validation], indices[num_validation:]
    X_train, X_val = train_data[train_indices], train_data[val_indices]
    y_train, y_val = train_labels[train_indices], train_labels[val_indices]
    return X_train, X_val, y_train, y_val


def initialize_weights(input_size, hidden_size, output_size):
    weights_input_hidden = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
    bias_hidden = np.zeros((1, hidden_size))
    weights_hidden_output = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
    bias_output = np.zeros((1, output_size))
    return weights_input_hidden, bias_hidden, weights_hidden_output, bias_output


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_derivative(z):
    return z * (1 - z)


def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / exp_z.sum(axis=1, keepdims=True)


def forward(X, weights_input_hidden, bias_hidden, weights_hidden_output, bias_output):
    z1 = np.dot(X, weights_input_hidden) + bias_hidden
    a1 = sigmoid(z1)
    z2 = np.dot(a1, weights_hidden_output) + bias_output
    a2 = softmax(z2)
    return a1, a2


def backward(X, y, a1, output, weights_hidden_output):
    output_error = output - y
    output_delta = output_error
    hidden_error = np.dot(output_delta, weights_hidden_output.T)
    hidden_delta = hidden_error * sigmoid_derivative(a1)
    return output_delta, hidden_delta


def update_weights(X, a1, output_delta, hidden_delta, weights_input_hidden, bias_hidden, weights_hidden_output, bias_output, learning_rate):
    weights_hidden_output -= learning_rate * np.dot(a1.T, output_delta)
    bias_output -= learning_rate * np.sum(output_delta, axis=0, keepdims=True)
    weights_input_hidden -= learning_rate * np.dot(X.T, hidden_delta)
    bias_hidden -= learning_rate * np.sum(hidden_delta, axis=0, keepdims=True)
    return weights_input_hidden, bias_hidden, weights_hidden_output, bias_output


def train(X, y, X_val, y_val, input_size=784, hidden_size=100, output_size=10, initial_learning_rate=0.01, decay_rate=0.95, epochs=10, batch_size=64, patience=2, threshold=0.01):
    weights_input_hidden, bias_hidden, weights_hidden_output, bias_output = initialize_weights(input_size, hidden_size, output_size)
    learning_rate = initial_learning_rate
    best_val_accuracy = 0
    epochs_no_improve = 0

    for epoch in range(epochs):
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        X, y = X[indices], y[indices]

        for i in range(0, X.shape[0], batch_size):
            X_batch = X[i:i + batch_size]
            y_batch = y[i:i + batch_size]
            a1, output = forward(X_batch, weights_input_hidden, bias_hidden, weights_hidden_output, bias_output)
            output_delta, hidden_delta = backward(X_batch, y_batch, a1, output, weights_hidden_output)
            weights_input_hidden, bias_hidden, weights_hidden_output, bias_output = update_weights(X_batch, a1, output_delta, hidden_delta, weights_input_hidden, bias_hidden, weights_hidden_output, bias_output, learning_rate)

        train_accuracy = evaluate(X, y, weights_input_hidden, bias_hidden, weights_hidden_output, bias_output)
        val_accuracy = evaluate(X_val, y_val, weights_input_hidden, bias_hidden, weights_hidden_output, bias_output)
        print(f"Epoch {epoch + 1}, Training Accuracy: {train_accuracy:.4f}, Validation Accuracy: {val_accuracy:.4f}")

        if val_accuracy > best_val_accuracy + threshold:
            best_val_accuracy = val_accuracy
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            learning_rate *= decay_rate
            print(f"Learning rate decayed to {learning_rate}")
            epochs_no_improve = 0
    return weights_input_hidden, bias_hidden, weights_hidden_output, bias_output


def evaluate(X, y, weights_input_hidden, bias_hidden, weights_hidden_output, bias_output):
    _, output = forward(X, weights_input_hidden, bias_hidden, weights_hidden_output, bias_output)
    predictions = np.argmax(output, axis=1)
    labels = np.argmax(y, axis=1)
    accuracy = np.mean(predictions == labels)
    return accuracy


def visualize_weights(W):
    num_classes = W.shape[1]
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))

    for i in range(num_classes):
        ax = axes[i // 5, i % 5]
        weight_image = W[:, i].reshape(28, 28)
        ax.imshow(weight_image, cmap='viridis')
        ax.set_title(f'Digit {i}')
        ax.axis('off')

    plt.tight_layout()
    plt.show()


X_train, X_val, y_train, y_val = load_mnist_with_validation()
weights_input_hidden, bias_hidden, weights_hidden_output, bias_output = train(X_train, y_train, X_val, y_val, epochs=30)
accuracy = evaluate(X_val, y_val, weights_input_hidden, bias_hidden, weights_hidden_output, bias_output)
print(f"Accuracy: {accuracy:.4f}")
visualize_weights(np.dot(weights_input_hidden, weights_hidden_output))
