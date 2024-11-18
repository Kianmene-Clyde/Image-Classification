#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <fstream>
#include <numeric>

class MLP {
public:
    MLP(const std::vector<int>& structure, double learning_rate, int epochs, bool is_classified)
        : structure(structure), learning_rate(learning_rate), epochs(epochs), is_classified(is_classified) {
        // Initialize weights and biases
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<> d(0, 1);
        
        for (size_t i = 0; i < structure.size() - 1; ++i) {
            std::vector<std::vector<double>> layer_weights(structure[i], std::vector<double>(structure[i + 1]));
            for (auto& node_weights : layer_weights) {
                for (auto& weight : node_weights) {
                    weight = d(gen);
                }
            }
            weights.push_back(layer_weights);
            biases.push_back(std::vector<double>(structure[i + 1], 0.0));
        }
    }

    void fit(const std::vector<std::vector<double>>& inputs, const std::vector<std::vector<double>>& labels) {
        for (int epoch = 0; epoch < epochs; ++epoch) {
            std::vector<std::vector<std::vector<double>>> summations(structure.size() - 1);
            std::vector<std::vector<std::vector<double>>> activations(structure.size() - 1);
            std::vector<std::vector<std::vector<double>>> errors(structure.size() - 1);
            std::vector<std::vector<std::vector<double>>> delta_weights(structure.size() - 1);
            std::vector<std::vector<double>> delta_biases(structure.size() - 1);

            // Feedforward
            for (size_t i = 0; i < weights.size(); ++i) {
                if (i == 0) {
                    summations[i] = matrix_add(matrix_dot(inputs, weights[i]), biases[i]);
                    activations[i] = sigmoid(summations[i]);
                } else {
                    summations[i] = matrix_add(matrix_dot(activations[i - 1], weights[i]), biases[i]);
                    activations[i] = sigmoid(summations[i]);
                }
            }

            // Backpropagation
            for (int i = weights.size() - 1; i >= 0; --i) {
                if (i == weights.size() - 1) {
                    errors[i] = matrix_subtract(activations[i], labels);
                    if (epoch % 100 == 0) {
                        std::cout << "Error: " << mse(errors[i]) << std::endl;
                    }
                    delta_weights[i] = matrix_dot(matrix_transpose(activations[i - 1]), matrix_multiply(errors[i], sigmoid_prime(summations[i])));
                    delta_biases[i] = sum_axis(matrix_multiply(errors[i], sigmoid_prime(summations[i])), 0);
                } else if (i == 0) {
                    errors[i] = matrix_multiply(matrix_dot(errors[i + 1], matrix_transpose(weights[i + 1])), sigmoid_prime(summations[i]));
                    delta_weights[i] = matrix_dot(matrix_transpose(inputs), errors[i]);
                    delta_biases[i] = sum_axis(errors[i], 0);
                } else {
                    errors[i] = matrix_multiply(matrix_dot(errors[i + 1], matrix_transpose(weights[i + 1])), sigmoid_prime(summations[i]));
                    delta_weights[i] = matrix_dot(matrix_transpose(activations[i - 1]), errors[i]);
                    delta_biases[i] = sum_axis(errors[i], 0);
                }
            }

            // Update weights and biases
            for (size_t i = 0; i < weights.size(); ++i) {
                weights[i] = matrix_subtract(weights[i], scalar_multiply(delta_weights[i], learning_rate));
                biases[i] = matrix_subtract(biases[i], scalar_multiply(delta_biases[i], learning_rate));
            }
        }
    }

    std::vector<std::vector<double>> predict(const std::vector<std::vector<double>>& inputs) {
        std::vector<std::vector<double>> summation, activation(inputs);

        for (size_t i = 0; i < weights.size(); ++i) {
            summation = matrix_add(matrix_dot(activation, weights[i]), biases[i]);
            activation = sigmoid(summation);
        }

        for (auto& act : activation) {
            for (auto& val : act) {
                val = val > 0.5 ? 1.0 : 0.0;
            }
        }

        return activation;
    }

    void save_data(const std::string& filename) {
        std::ofstream file(filename, std::ios::binary);
        if (file.is_open()) {
            size_t structure_size = structure.size();
            file.write(reinterpret_cast<const char*>(&structure_size), sizeof(structure_size));
            file.write(reinterpret_cast<const char*>(structure.data()), structure.size() * sizeof(int));
            file.write(reinterpret_cast<const char*>(&learning_rate), sizeof(learning_rate));
            file.write(reinterpret_cast<const char*>(&epochs), sizeof(epochs));
            file.write(reinterpret_cast<const char*>(&is_classified), sizeof(is_classified));

            for (const auto& layer_weights : weights) {
                size_t rows = layer_weights.size();
                size_t cols = layer_weights[0].size();
                file.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
                file.write(reinterpret_cast<const char*>(&cols), sizeof(cols));
                for (const auto& node_weights : layer_weights) {
                    file.write(reinterpret_cast<const char*>(node_weights.data()), node_weights.size() * sizeof(double));
                }
            }

            for (const auto& layer_biases : biases) {
                size_t size = layer_biases.size();
                file.write(reinterpret_cast<const char*>(&size), sizeof(size));
                file.write(reinterpret_cast<const char*>(layer_biases.data()), layer_biases.size() * sizeof(double));
            }

            file.close();
        }
    }

    void load_data(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary);
        if (file.is_open()) {
            size_t structure_size;
            file.read(reinterpret_cast<char*>(&structure_size), sizeof(structure_size));
            structure.resize(structure_size);
            file.read(reinterpret_cast<char*>(structure.data()), structure.size() * sizeof(int));
            file.read(reinterpret_cast<char*>(&learning_rate), sizeof(learning_rate));
            file.read(reinterpret_cast<char*>(&epochs), sizeof(epochs));
            file.read(reinterpret_cast<char*>(&is_classified), sizeof(is_classified));

            weights.clear();
            biases.clear();

            for (size_t i = 0; i < structure_size - 1; ++i) {
                size_t rows, cols;
                file.read(reinterpret_cast<char*>(&rows), sizeof(rows));
                file.read(reinterpret_cast<char*>(&cols), sizeof(cols));
                std::vector<std::vector<double>> layer_weights(rows, std::vector<double>(cols));
                for (auto& node_weights : layer_weights) {
                    file.read(reinterpret_cast<char*>(node_weights.data()), node_weights.size() * sizeof(double));
                }
                weights.push_back(layer_weights);
            }

            for (size_t i = 0; i < structure_size - 1; ++i) {
                size_t size;
                file.read(reinterpret_cast<char*>(&size), sizeof(size));
                std::vector<double> layer_biases(size);
                file.read(reinterpret_cast<char*>(layer_biases.data()), layer_biases.size() * sizeof(double));
                biases.push_back(layer_biases);
            }

            file.close();
        }
    }

private:
    std::vector<int> structure;
    double learning_rate;
    int epochs;
    bool is_classified;
    std::vector<std::vector<std::vector<double>>> weights;
    std::vector<std::vector<double>> biases;

    static std::vector<std::vector<double>> sigmoid(const std::vector<std::vector<double>>& x) {
        std::vector<std::vector<double>> result(x.size(), std::vector<double>(x[0].size()));
        for (size_t i = 0; i < x.size(); ++i) {
            for (size_t j = 0; j < x[i].size(); ++j) {
                result[i][j] = 1.0 / (1.0 + exp(-x[i][j]));
            }
        }
        return result;
    }

    static std::vector<std::vector<double>> sigmoid_prime(const std::vector<std::vector<double>>& x) {
        std::vector<std::vector<double>> result(x.size(), std::vector<double>(x[0].size()));
        for (size_t i = 0; i < x.size(); ++i) {
            for (size_t j = 0; j < x[i].size(); ++j) {
                result[i][j] = x[i][j] * (1.0 - x[i][j]);
            }
        }
        return result;
    }

    static std::vector<std::vector<double>> matrix_dot(const std::vector<std::vector<double>>& a, const std::vector<std::vector<double>>& b) {
        std::vector<std::vector<double>> result(a.size(), std::vector<double>(b[0].size()));
        for (size_t i = 0; i < a.size(); ++i) {
            for (size_t j = 0; j < b[0].size(); ++j) {
                result[i][j] = 0;
                for (size_t k = 0; k < b.size(); ++k) {
                    result[i][j] += a[i][k] * b[k][j];
                }
            }
        }
        return result;
    }

    static std::vector<std::vector<double>> matrix_add(const std::vector<std::vector<double>>& a, const std::vector<double>& b) {
        std::vector<std::vector<double>> result(a.size(), std::vector<double>(a[0].size()));
        for (size_t i = 0; i < a.size(); ++i) {
            for (size_t j = 0; j < a[i].size(); ++j) {
                result[i][j] = a[i][j] + b[j];
            }
        }
        return result;
    }

    static std::vector<std::vector<double>> matrix_subtract(const std::vector<std::vector<double>>& a, const std::vector<std::vector<double>>& b) {
        std::vector<std::vector<double>> result(a.size(), std::vector<double>(a[0].size()));
        for (size_t i = 0; i < a.size(); ++i) {
            for (size_t j = 0; j < a[i].size(); ++j) {
                result[i][j] = a[i][j] - b[i][j];
            }
        }
        return result;
    }

    static std::vector<std::vector<double>> scalar_multiply(const std::vector<std::vector<double>>& a, double scalar) {
        std::vector<std::vector<double>> result(a.size(), std::vector<double>(a[0].size()));
        for (size_t i = 0; i < a.size(); ++i) {
            for (size_t j = 0; j < a[i].size(); ++j) {
                result[i][j] = a[i][j] * scalar;
            }
        }
        return result;
    }

    static std::vector<double> sum_axis(const std::vector<std::vector<double>>& a, int axis) {
        std::vector<double> result;
        if (axis == 0) {
            result.resize(a[0].size(), 0.0);
            for (const auto& row : a) {
                for (size_t j = 0; j < row.size(); ++j) {
                    result[j] += row[j];
                }
            }
        } else if (axis == 1) {
            result.resize(a.size(), 0.0);
            for (size_t i = 0; i < a.size(); ++i) {
                result[i] = std::accumulate(a[i].begin(), a[i].end(), 0.0);
            }
        }
        return result;
    }

    static std::vector<std::vector<double>> matrix_transpose(const std::vector<std::vector<double>>& a) {
        std::vector<std::vector<double>> result(a[0].size(), std::vector<double>(a.size()));
        for (size_t i = 0; i < a.size(); ++i) {
            for (size_t j = 0; j < a[i].size(); ++j) {
                result[j][i] = a[i][j];
            }
        }
        return result;
    }

    static double mse(const std::vector<std::vector<double>>& errors) {
        double sum = 0.0;
        for (const auto& row : errors) {
            for (double val : row) {
                sum += val * val;
            }
        }
        return sum / errors.size();
    }
};