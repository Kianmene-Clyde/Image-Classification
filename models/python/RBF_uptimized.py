import json
import numpy as np
from sklearn.cluster import KMeans
import tensorflow as tf


class RBF:
    def __init__(self, num_of_classes, k_num_clusters, gamma, epochs, std_from_clusters=True):
        self.num_of_classes = num_of_classes
        self.k_num_clusters = k_num_clusters
        self.gamma = gamma
        self.epochs = epochs
        self.std_from_clusters = std_from_clusters
        self.weights = None
        self.centroids = None
        self.std_list = None

    @staticmethod
    def one_hot_encode(labels, num_classes):
        encoded_labels = np.zeros((len(labels), num_classes))
        for index, label in enumerate(labels):
            encoded_labels[index, label] = 1
        return encoded_labels

    def get_rbf(self, input_neuron, centroid):
        distance = np.linalg.norm(input_neuron - centroid)
        return np.exp(-distance * self.gamma)

    def get_rbf_as_list(self, input_vector, centroids, std_list):
        RBF_list = []
        for neuron in input_vector:
            RBF_list.append([self.get_rbf(neuron, centroid) for (centroid, std) in zip(centroids, std_list)])
        return np.array(RBF_list)

    def fit(self, training_inputs, training_labels, test_inputs=None, test_labels=None, logdir=None):
        print("Training Started...")

        if logdir:
            writer = tf.summary.create_file_writer(logdir)

        kmeans = KMeans(n_clusters=self.k_num_clusters, max_iter=self.epochs)
        kmeans.fit(training_inputs)
        self.centroids = kmeans.cluster_centers_

        if self.std_from_clusters:
            self.std_list = np.std(kmeans.transform(training_inputs), axis=0)
        else:
            d_max = np.max([np.linalg.norm(c1 - c2) for c1 in self.centroids for c2 in self.centroids])
            self.std_list = np.repeat(d_max / np.sqrt(2 * self.k_num_clusters), self.k_num_clusters)

        for epoch in range(self.epochs):
            RBF_X = self.get_rbf_as_list(training_inputs, self.centroids, self.std_list)
            self.weights = np.linalg.pinv(RBF_X.T @ RBF_X) @ RBF_X.T @ self.one_hot_encode(training_labels,
                                                                                           self.num_of_classes)

            training_predictions = RBF_X @ self.weights
            training_predictions = np.array([np.argmax(label) for label in training_predictions])
            train_accuracy = np.mean(training_predictions == training_labels)
            train_loss = np.mean((training_predictions - training_labels) ** 2)

            if test_inputs is not None and test_labels is not None:
                test_predictions = self.predict_all(test_inputs)
                test_accuracy = np.mean(test_predictions == test_labels)
                test_loss = np.mean((test_predictions - test_labels) ** 2)

            if logdir:
                with writer.as_default():
                    tf.summary.scalar('train_loss', train_loss, step=epoch)
                    tf.summary.scalar('train_accuracy', train_accuracy, step=epoch)

                    if test_inputs is not None and test_labels is not None:
                        tf.summary.scalar('test_loss', test_loss, step=epoch)
                        tf.summary.scalar('test_accuracy', test_accuracy, step=epoch)
                    writer.flush()

            # print(f"Epoch {epoch + 1}/{self.epochs}, Train Loss: {train_loss}, Train Accuracy: {train_accuracy}")

        print(f"Training Accuracy: {train_accuracy * 100}%")
        print(f"Test Accuracy: {test_accuracy * 100}%")

        print("Training Complete...")

    def predict_all(self, inputs):
        RBF_X = self.get_rbf_as_list(inputs, self.centroids, self.std_list)
        predictions = RBF_X @ self.weights
        predictions = np.array([np.argmax(label) for label in predictions])

        return predictions

    def predict(self, input):
        input = np.array([input])
        RBF_X = self.get_rbf_as_list(input, self.centroids, self.std_list)
        prediction = RBF_X @ self.weights
        prediction = np.argmax(prediction[0])

        return prediction

    def save_data_json(self, filename):
        data = {
            'num_of_classes': self.num_of_classes,
            'k_num_clusters': self.k_num_clusters,
            'gamma': self.gamma,
            'epochs': self.epochs,
            'weights': self.weights.tolist() if self.weights is not None else None,
            'centroids': self.centroids.tolist() if self.centroids is not None else None,
            'std_list': self.std_list.tolist() if self.std_list is not None else None
        }
        with open(filename, 'w') as file:
            json.dump(data, file)

    def load_data_json(self, filename):
        with open(filename, 'r') as file:
            data = json.load(file)
            self.num_of_classes = data['num_of_classes']
            self.k_num_clusters = data['k_num_clusters']
            self.gamma = data['gamma']
            self.epochs = data['epochs']
            self.weights = np.array(data['weights']) if data['weights'] is not None else None
            self.centroids = np.array(data['centroids']) if data['centroids'] is not None else None
            self.std_list = np.array(data['std_list']) if data['std_list'] is not None else None
