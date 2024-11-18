import json
import numpy as np
import tensorflow as tf


def get_distance(first_point_list, second_point_list):
    return np.sum((first_point_list - second_point_list) ** 2)


def kmeans(input_data, k_num_clusters, max_iters):
    centroids = input_data[np.random.choice(range(len(input_data)), k_num_clusters, replace=False)]
    converged = False
    current_iter = 0

    while not converged and current_iter < max_iters:
        cluster_list = [[] for _ in range(len(centroids))]

        # on calcule la distance entre chaque echantillons et chaque centroids et on affecte l'echantillon dans le cluster
        # avec la distance minimal
        for data in input_data:
            distances = [get_distance(c, data) for c in centroids]
            cluster_index = int(np.argmin(distances))
            cluster_list[cluster_index].append(data)

        prev_centroids = centroids.copy()
        centroids = np.array(
            [np.mean(cluster, axis=0) if cluster else c for c, cluster in zip(centroids, cluster_list)])

        difference = np.sum(np.abs(prev_centroids - centroids))
        converged = (difference < 1e-6)
        current_iter += 1

    return np.array(centroids), [np.std(cluster) for cluster in cluster_list]


class RBF:
    def __init__(self, num_of_classes, k_num_clusters, learning_rate, gamma, epochs, batch_size=64,
                 std_from_clusters=True, is_classified=True):
        self.num_of_classes = num_of_classes
        self.k_num_clusters = k_num_clusters
        self.gamma = gamma
        self.epochs = epochs
        self.batch_size = batch_size
        self.std_from_clusters = std_from_clusters
        self.is_classified = is_classified
        self.learning_rate = learning_rate
        self.weights = None
        self.centroids = None
        self.std_list = None

    @staticmethod
    def one_hot_encode(labels, num_classes):
        return np.eye(num_classes)[labels]

    def get_rbf(self, input_neuronne, centroid):
        distance = get_distance(input_neuronne, centroid)
        return np.exp(-self.gamma * distance)

    def get_rbf_as_list(self, input_vector, centroids):
        return np.array([[self.get_rbf(neuronne, centroid) for centroid in centroids] for neuronne in input_vector])

    def compute_loss(self, predictions, labels):
        if self.is_classified:
            loss = np.mean(np.sum(-labels * np.log(predictions + 1e-9), axis=1))  # + 1e-9 pour eviter les NANs
        else:
            loss = np.mean((predictions.flatten() - labels.flatten()) ** 2)
        return loss

    def softmax(self, prediction):
        e_x = np.exp(prediction - np.max(prediction, axis=1, keepdims=True))
        return e_x / e_x.sum(axis=1, keepdims=True)

    def fit(self, training_inputs, training_labels, test_inputs=None, test_labels=None, logdir=None):
        print("Training Started...")

        if logdir:
            writer = tf.summary.create_file_writer(logdir)

        self.centroids, self.std_list = kmeans(training_inputs, self.k_num_clusters, self.epochs)

        if not self.std_from_clusters:
            dMax = np.max([get_distance(c1, c2) for c1 in self.centroids for c2 in self.centroids])
            self.std_list = np.repeat(dMax / np.sqrt(2 * self.k_num_clusters), self.k_num_clusters)

        num_samples = training_inputs.shape[0]

        if self.is_classified:
            self.weights = np.random.randn(self.k_num_clusters, self.num_of_classes)
            training_labels = self.one_hot_encode(training_labels, self.num_of_classes)
            if test_labels is not None:
                test_labels = self.one_hot_encode(test_labels, self.num_of_classes)
        else:
            self.weights = np.random.randn(self.k_num_clusters)

        for epoch in range(self.epochs):

            indices = np.random.permutation(num_samples)
            training_inputs = training_inputs[indices]
            training_labels = training_labels[indices]

            for i in range(0, num_samples, self.batch_size):
                batch_inputs = training_inputs[i:i + self.batch_size]
                batch_labels = training_labels[i:i + self.batch_size]

                RBF_X = self.get_rbf_as_list(batch_inputs, self.centroids)
                batch_predictions = RBF_X @ self.weights

                if self.is_classified:
                    batch_predictions = self.softmax(batch_predictions)
                    errors = batch_labels - batch_predictions
                    self.weights += self.learning_rate * RBF_X.T @ errors / self.batch_size
                else:
                    errors = batch_labels - batch_predictions.flatten()
                    self.weights += self.learning_rate * RBF_X.T @ errors / self.batch_size

            # Calculate and log loss and accuracy
            RBF_X_train = self.get_rbf_as_list(training_inputs, self.centroids)
            train_predictions = RBF_X_train @ self.weights
            if self.is_classified:
                train_predictions = self.softmax(train_predictions)
            train_loss = self.compute_loss(train_predictions, training_labels)
            train_accuracy = np.mean(np.argmax(train_predictions, axis=1) == np.argmax(training_labels, axis=1))

            if test_inputs is not None and test_labels is not None:
                RBF_X_test = self.get_rbf_as_list(test_inputs, self.centroids)
                test_predictions = RBF_X_test @ self.weights
                if self.is_classified:
                    test_predictions = self.softmax(test_predictions)
                test_loss = self.compute_loss(test_predictions, test_labels)
                test_accuracy = np.mean(np.argmax(test_predictions, axis=1) == np.argmax(test_labels, axis=1))
            else:
                test_loss = None
                test_accuracy = None

            if logdir:
                with writer.as_default():
                    tf.summary.scalar('train_loss', train_loss, step=epoch)
                    tf.summary.scalar('train_accuracy', train_accuracy, step=epoch)
                    if test_inputs is not None and test_labels is not None:
                        tf.summary.scalar('test_loss', test_loss, step=epoch)
                        tf.summary.scalar('test_accuracy', test_accuracy, step=epoch)
                    writer.flush()

        print(f"Train Accuracy: {train_accuracy * 100} %")
        if test_inputs is not None and test_labels is not None:
            print(f"Test Accuracy: {test_accuracy * 100} %")

        print("Training Complete...")

    def predict_all(self, inputs):
        RBF_X = self.get_rbf_as_list(inputs, self.centroids)
        predictions = RBF_X @ self.weights
        if self.is_classified:
            predictions = self.softmax(predictions)
            predictions = np.argmax(predictions, axis=1)
        return predictions

    def predict(self, input):
        input = np.array([input])
        RBF_X = self.get_rbf_as_list(input, self.centroids)
        prediction = RBF_X @ self.weights
        if self.is_classified:
            prediction = self.softmax(prediction)
            prediction = np.argmax(prediction[0])
        return prediction

    def save_data_json(self, filename):
        data = {
            'num_of_classes': self.num_of_classes,
            'k_num_clusters': self.k_num_clusters,
            'learning_rate': self.learning_rate,
            'gamma': self.gamma,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'is_classified': self.is_classified,
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
            self.learning_rate = data['learning_rate']
            self.gamma = data['gamma']
            self.epochs = data['epochs']
            self.batch_size = data['batch_size']
            self.is_classified = data['is_classified']
            self.weights = np.array(data['weights']) if data['weights'] is not None else None
            self.centroids = np.array(data['centroids']) if data['centroids'] is not None else None
            self.std_list = np.array(data['std_list']) if data['std_list'] is not None else None
