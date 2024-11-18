import numpy as np
import json
import tensorflow as tf


class SVM:
    def __init__(self, learning_rate=0.01, lambda_param=0.01, epochs=1000, num_classes=3, batch_size=64):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.epochs = epochs
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.weights = None
        self.bias = None

    def init_weights_bias(self, inputs):
        num_features = inputs.shape[1]
        self.weights = np.random.rand(self.num_classes, num_features)
        self.bias = np.ones(self.num_classes)

    def softmax(self, prediction):
        exp_x = np.exp(prediction - np.max(prediction, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def loss_calculation(self, inputs, labels):
        num_samples = inputs.shape[0]
        linear_model = np.dot(inputs, self.weights.T) + self.bias
        probabilities = self.softmax(linear_model)
        log = -np.log(probabilities[range(num_samples), labels] + 1e-15)
        loss = np.sum(log) / num_samples
        loss += self.lambda_param * np.sum(self.weights ** 2) / 2
        return loss

    def gradient(self, inputs, labels):
        num_samples = inputs.shape[0]
        linear_model = np.dot(inputs, self.weights.T) + self.bias
        probabilities = self.softmax(linear_model)
        probabilities[range(num_samples), labels] -= 1
        delta_weights = np.dot(probabilities.T, inputs) / num_samples
        delta_bias = np.sum(probabilities, axis=0) / num_samples
        delta_weights += self.lambda_param * self.weights

        predictions = np.argmax(linear_model, axis=1)
        accuracy = np.mean(predictions == labels)

        return delta_weights, delta_bias, accuracy

    def fit(self, training_inputs, training_labels, test_inputs=None, test_labels=None, logdir=None):
        print("Training Started")

        self.init_weights_bias(training_inputs)
        num_samples = training_inputs.shape[0]

        if logdir:
            writer = tf.summary.create_file_writer(logdir)

        for epoch in range(self.epochs):
            indices = np.random.permutation(num_samples)
            training_inputs = training_inputs[indices]
            training_labels = training_labels[indices]

            for i in range(0, num_samples, self.batch_size):
                batch_inputs = training_inputs[i:i + self.batch_size]
                batch_labels = training_labels[i:i + self.batch_size]

                delta_weights, delta_bias, train_accuracy = self.gradient(batch_inputs, batch_labels)
                self.weights -= self.learning_rate * delta_weights
                self.bias -= self.learning_rate * delta_bias

            train_loss = self.loss_calculation(training_inputs, training_labels)

            if test_inputs is not None and test_labels is not None:
                test_predictions = self.predict_all(test_inputs)
                test_loss = self.loss_calculation(test_inputs, test_labels)
                test_accuracy = np.mean(test_predictions == test_labels)

            if logdir:
                with writer.as_default():
                    tf.summary.scalar('train_loss', train_loss, step=epoch)
                    tf.summary.scalar('train_accuracy', train_accuracy, step=epoch)

                    if test_inputs is not None and test_labels is not None:
                        tf.summary.scalar('test_loss', test_loss, step=epoch)
                        tf.summary.scalar('test_accuracy', test_accuracy, step=epoch)
                    writer.flush()

            # print(f"Epoch {epoch + 1}/{self.epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")

        print(f"Training Accuracy: {train_accuracy * 100} %")
        if test_inputs is not None and test_labels is not None:
            print(f"Test Accuracy: {test_accuracy * 100} %")
        print("Training finished...")

    def predict_all(self, inputs):
        linear_model = np.dot(inputs, self.weights.T) + self.bias
        probabilities = self.softmax(linear_model)
        predictions = np.argmax(probabilities, axis=1)
        return predictions

    def predict(self, input):
        input = input.reshape(1, -1)
        linear_model = np.dot(input, self.weights.T) + self.bias
        probabilities = self.softmax(linear_model)
        prediction = np.argmax(probabilities, axis=1)
        return prediction[0]

    def save_data_json(self, filename):
        data = {
            'learning_rate': self.learning_rate,
            'lambda_param': self.lambda_param,
            'epochs': self.epochs,
            'num_classes': self.num_classes,
            'weights': self.weights.tolist(),
            'bias': self.bias.tolist()
        }
        with open(filename, 'w') as file:
            json.dump(data, file)

    def load_data_json(self, filename):
        with open(filename, 'r') as file:
            data = json.load(file)
            self.learning_rate = data['learning_rate']
            self.lambda_param = data['lambda_param']
            self.epochs = data['epochs']
            self.num_classes = data['num_classes']
            self.weights = np.array(data['weights'])
            self.bias = np.array(data['bias'])
