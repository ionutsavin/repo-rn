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
    dataset = MNIST(root='data/', transform=lambda x: np.array(x).flatten(), train=is_train, download=False)
    mnist_data = []
    mnist_labels = []
    for image, label in dataset:
        mnist_data.append(image)
        mnist_labels.append(label)

    mnist_data = normalize_data(mnist_data)
    mnist_labels = one_hot_encode(mnist_labels)

    return mnist_data, mnist_labels


def create_batches(X, y, batch_size=100):
    num_batches = len(X) // batch_size
    batches_X = np.array_split(X, num_batches)
    batches_y = np.array_split(y, num_batches)
    return batches_X, batches_y


def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # numerical stability
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def compute_loss(predictions, labels):
    loss = -np.sum(labels * np.log(predictions)) / labels.shape[0]
    return loss


def gradient_descent(X_batch, y_batch, W, b, learning_rate):
    num_samples = X_batch.shape[0]

    # forward propagation
    z = np.dot(X_batch, W) + b
    predictions = softmax(z)

    # updating weights and biases
    error = predictions - y_batch
    gradient_W = np.dot(X_batch.T, error) / num_samples
    gradient_b = np.sum(error, axis=0) / num_samples
    W -= learning_rate * gradient_W
    b -= learning_rate * gradient_b

    return W, b


def train_perceptron(train_X, train_y, num_epochs=50, batch_size=100, learning_rate=0.01):
    num_features = train_X.shape[1]
    num_classes = train_y.shape[1]
    W = np.random.randn(num_features, num_classes) * 0.01
    b = np.zeros(num_classes)

    for epoch in range(num_epochs):
        indices = np.random.permutation(train_X.shape[0])
        shuffled_X = train_X[indices]
        shuffled_y = train_y[indices]
        batches_X, batches_y = create_batches(shuffled_X, shuffled_y, batch_size)

        total_loss = 0
        for X_batch, y_batch in zip(batches_X, batches_y):
            W, b = gradient_descent(X_batch, y_batch, W, b, learning_rate)

            z = np.dot(X_batch, W) + b
            predictions = softmax(z)
            total_loss += compute_loss(predictions, y_batch)

        avg_loss = total_loss / len(batches_X)
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}')

    return W, b


def compute_accuracy(X, y, W, b):
    z = np.dot(X, W) + b
    predictions = softmax(z)
    predicted_labels = np.argmax(predictions, axis=1)
    true_labels = np.argmax(y, axis=1)

    accuracy = np.mean(predicted_labels == true_labels)
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


train_X, train_Y = load_mnist(True)
test_X, test_Y = load_mnist(False)
W, b = train_perceptron(train_X, train_Y, num_epochs=100, batch_size=100, learning_rate=0.01)
accuracy = compute_accuracy(test_X, test_Y, W, b)

print(f'Test Accuracy: {accuracy * 100:.2f}%')
visualize_weights(W)
