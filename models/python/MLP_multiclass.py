import json
import numpy as np
import tensorflow as tf


class MLP:
    def __init__(self, structure: list, learning_rate: float = 0.01, epochs: int = 1000,
                 is_classification: bool = True) -> None:
        self.structure = structure
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.is_classification = is_classification

        self.biases = [np.zeros((1, layer_size)) for layer_size in self.structure[1:]]
        self.weights = [np.random.randn(self.structure[i], self.structure[i + 1]) * np.sqrt(2.0 / self.structure[i])
                        for i in range(len(self.structure) - 1)]

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum(axis=1, keepdims=True)

    def mse(self, label, prediction):
        return np.mean(np.square(label - prediction))

    def feed_forward(self, input):

        activations = [None] * len(self.weights)
        summation = [None] * len(self.weights)

        for i in range(len(self.weights)):
            if i == 0:
                summation[i] = np.dot(input.T, self.weights[i]) + self.biases[i]
            else:
                summation[i] = np.dot(activations[i - 1], self.weights[i]) + self.biases[i]
            if self.is_classification:
                activations[i] = self.softmax(summation[i]) if i == len(summation) - 1 else np.tanh(summation[i])
            else:
                activations[i] = summation[i]

        return activations

    def fit(self, training_inputs, training_labels, test_inputs=None, test_labels=None, logdir=None):
        print("Training started")

        training_inputs = np.array(training_inputs)
        training_labels = np.array(training_labels)

        if logdir:
            writer = tf.summary.create_file_writer(logdir)

        for epoch in range(self.epochs):
            train_loss = 0.0
            test_loss = 0.0
            train_accuracies = []
            test_accuracies = []

            for input, label in zip(training_inputs, training_labels):
                input = input.reshape(-1, 1)
                label = label.reshape(1, -1)

                errors = [None] * len(self.weights)

                activations = self.feed_forward(input)

                # Calcul des erreurs et de la perte
                errors[-1] = activations[-1] - label
                train_loss += self.mse(label, activations[-1])

                # Backpropagation
                for l in range(len(errors) - 2, -1, -1):
                    errors[l] = np.dot(errors[l + 1] * (1 - activations[l + 1] ** 2), self.weights[l + 1].T)

                # Mise à jour des poids et des biais
                for l in range(len(self.weights)):
                    self.weights[l] -= self.learning_rate * np.dot((input if l == 0 else activations[l - 1].T),
                                                                   errors[l])
                    self.biases[l] -= self.learning_rate * np.sum(errors[l], axis=0, keepdims=True)

                train_prediction = self.predict(input)
                if self.is_classification:
                    train_prediction = np.argmax(train_prediction)
                    train_label = np.argmax(label)
                    train_accuracies.append(train_prediction == train_label)

            train_loss /= len(training_inputs)
            train_accuracy = np.mean(train_accuracies)

            if test_inputs is not None and test_labels is not None:
                for input, label in zip(test_inputs, test_labels):
                    input = input.reshape(-1, 1)
                    label = label.reshape(1, -1)

                    activations = self.feed_forward(input)
                    if self.is_classification:
                        test_prediction = np.argmax(activations[-1])
                        test_label = np.argmax(label)
                        test_accuracies.append(test_prediction == test_label)
                    else:
                        test_prediction = activations[-1]

                    test_loss += self.mse(label, test_prediction)
                test_loss /= len(test_inputs)
                test_accuracy = np.mean(test_accuracies)

            if logdir:
                with writer.as_default():
                    tf.summary.scalar('train_loss', train_loss, step=epoch)
                    tf.summary.scalar('train_accuracy', train_accuracy, step=epoch)

                    if test_inputs is not None and test_labels is not None:
                        tf.summary.scalar('test_loss', test_loss, step=epoch)
                        tf.summary.scalar('test_accuracy', test_accuracy, step=epoch)
                    writer.flush()

            # print(f"Epoch {epoch + 1}/{self.epochs}, Train Loss: {train_loss}, Train Accuracy: {train_accuracy}")

        print(f"Train Accuracy: {train_accuracy * 100} %")
        if test_inputs is not None and test_labels is not None:
            print(f"Test Accuracy: {test_accuracy * 100} %")

        print("Training complete")

    def predict(self, input):
        activations = self.feed_forward(input)
        return np.argmax(activations[-1], axis=1)[0] if self.is_classification else activations[-1]

    def save_data_json(self, filename):
        data = {
            'structure': self.structure,
            'weights': [w.tolist() for w in self.weights],
            'biases': [b.tolist() for b in self.biases],
            'learning_rate': self.learning_rate,
            'epochs': self.epochs
        }
        with open(filename, 'w') as file:
            json.dump(data, file)

    def load_data_json(self, filename):
        with open(filename, 'r') as file:
            data = json.load(file)
            self.structure = data['structure']
            self.weights = [np.array(w) for w in data['weights']]
            self.biases = [np.array(b) for b in data['biases']]
            self.learning_rate = data['learning_rate']
            self.epochs = data['epochs']
