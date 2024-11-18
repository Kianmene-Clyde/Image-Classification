import pickle
import numpy as np


class LinearModel:
    def __init__(self, learning_rate: float = 0.01, epochs: int = 1000, is_classification: bool = True) -> None:
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.is_classification = is_classification
        self.weights = None
        self.biais = 0

    def mse(self, label, prediction):
        mse = np.mean((label - prediction) ** 2)
        return mse

    def train(self, inputs, labels) -> None:
        inputs = np.array(inputs)
        labels = np.array(labels)

        mean = np.mean(inputs, axis=0)
        std = np.std(inputs, axis=0)
        inputs = (inputs - mean) / std

        self.weights = np.random.randn(inputs.shape[1]) * np.sqrt(1 / inputs.shape[1])

        if self.is_classification:
            accuracy = 0.0
            error = 0.0

            for epoch in range(self.epochs):
                predictions = np.tanh(np.dot(inputs, self.weights.T) + self.biais)
                for i in range(inputs.shape[0]):
                    if labels[i] * (np.dot(inputs[i], self.weights.T) + self.biais) <= 0:
                        self.weights += self.learning_rate * (labels[i] - predictions[i]) * inputs[i]
                        self.biais += self.learning_rate * (labels[i] - predictions[i])

                if epoch % 100 == 0:
                    print("Error: ", error)

                accuracy = np.mean(np.where(predictions > 0.0, 1, -1) == labels)
            print("Accuracy: ", accuracy * 100, '%')
        else:
            inputs_with_bias = np.hstack((np.ones((inputs.shape[0], 1)), inputs))
            weights_biais_correction = np.dot(np.linalg.pinv(inputs_with_bias), labels)
            self.weights = weights_biais_correction[1:]
            self.biais = weights_biais_correction[0]
            predictions = np.dot(inputs, self.weights.T) + self.biais
            errors = self.mse(labels, predictions)

        print("training finished")

    def predict(self, inputs):
        inputs = np.array(inputs)
        mean = np.mean(inputs, axis=0)
        std = np.std(inputs, axis=0)
        inputs = (inputs - mean) / std
        if self.is_classification:
            predictions = np.tanh(np.dot(inputs, self.weights.T) + self.biais)
            return np.where(predictions > 0.0, 1, -1)
        else:
            predictions = np.dot(inputs, self.weights.T) + self.biais
            return predictions

    def save_data(self, filename):
        data = {
            'weights': self.weights,
            'bias': self.biais,
            'learning_rate': self.learning_rate,
            'epochs': self.epochs,
            'is_classified': self.is_classification
        }
        with open(filename, 'wb') as file:
            pickle.dump(data, file)

    def load_data(self, filename):
        with open(filename, 'rb') as file:
            data = pickle.load(file)
            self.weights = data['weights']
            self.biais = data['bias']
            self.learning_rate = data['learning_rate']
            self.epochs = data['epochs']
            self.is_classification = data['is_classified']
