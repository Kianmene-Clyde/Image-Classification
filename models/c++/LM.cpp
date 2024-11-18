#include <iostream>
#include <cstdint>
#include <vector>
#include <cmath>
#include <numeric>
#include <fstream>
#include <cstdlib>
#include <ctime>
#include <random>

#ifdef WIN32
#define DLLEXPORT __declspec(dllexport)
#else
#define DLLEXPORT
#endif

extern "C" {
    class DLLEXPORT LinearModel {
    private:
        std::vector<std::vector<double>> inputs;
        std::vector<double> labels;
        std::vector<double> weights;
        double bias;
        double learning_rate;
        int epochs;
        bool is_classified;

        double static sigmoid(double x) {
            return 0.5 * (1 + std::tanh(0.5 * x));
        }

        double static mse(std::vector<double>& labels, const std::vector<double>& predictions) {
            double sum = 0.0;
            for (size_t i = 0; i < labels.size(); ++i) {
                sum += std::pow(labels[i] - predictions[i], 2);
            }
            return sum / labels.size();
        }

        double static categorical_crossentropy(std::vector<double>& labels, const std::vector<double>& predictions) {
            double sum = 0.0;
            for (size_t i = 0; i < labels.size(); ++i) {
                double clipped_pred = std::min(std::max(predictions[i], 1e-7), 1.0 - 1e-7);
                sum += -labels[i] * std::log(clipped_pred);
            }
            return sum;
        }

        static std::vector<double> matrix_vector_product(const std::vector<std::vector<double>>& matrix, const std::vector<double>& vec, double bias) {
            std::vector<double> result(matrix.size());
            for (size_t i = 0; i < matrix.size(); ++i) {
                result[i] = std::inner_product(matrix[i].begin(), matrix[i].end(), vec.begin(), bias);
            }
            return result;
        }

        static std::vector<double> vector_subtract(const std::vector<double>& a, const std::vector<double>& b) {
            std::vector<double> result(a.size());
            for (size_t i = 0; i < a.size(); ++i) {
                result[i] = a[i] - b[i];
            }
            return result;
        }

        static std::vector<double> scalar_product(const std::vector<double>& vec, double scalar) {
            std::vector<double> result(vec.size());
            for (size_t i = 0; i < vec.size(); ++i) {
                result[i] = vec[i] * scalar;
            }
            return result;
        }

        static std::vector<double> vector_add(const std::vector<double>& a, const std::vector<double>& b) {
            std::vector<double> result(a.size());
            for (size_t i = 0; i < a.size(); ++i) {
                result[i] = a[i] + b[i];
            }
            return result;
        }

        static std::vector<double> matrix_transpose_product(const std::vector<std::vector<double>>& matrix, const std::vector<double>& vec) {
            size_t rows = matrix.size();
            size_t cols = matrix[0].size();
            std::vector<double> result(cols, 0.0);
            for (size_t i = 0; i < rows; ++i) {
                for (size_t j = 0; j < cols; ++j) {
                    result[j] += matrix[i][j] * vec[i];
                }
            }
            return result;
        }

    public:
        LinearModel(const std::vector<std::vector<double>>& inputs, const std::vector<double>& labels, double learning_rate = 0.01, int epochs = 1000, bool is_classified = true)
                : inputs(inputs), labels(labels), weights(std::vector<double>()), learning_rate(learning_rate), epochs(epochs), is_classified(is_classified) {
            bias = 0.0;
        }

        void train(const std::vector<std::vector<double>>& inputs, const std::vector<double>& labels) {
            std::vector<double> mean(inputs[0].size(), 0.0);
            std::vector<double> std(inputs[0].size(), 0.0);
            for (const auto& input : inputs) {
                for (size_t i = 0; i < input.size(); ++i) {
                    mean[i] += input[i];
                }
            }
            for (size_t i = 0; i < mean.size(); ++i) {
                mean[i] /= inputs.size();
            }
            for (const auto& input : inputs) {
                for (size_t i = 0; i < input.size(); ++i) {
                    std[i] += std::pow(input[i] - mean[i], 2);
                }
            }
            for (size_t i = 0; i < std.size(); ++i) {
                std[i] = std::sqrt(std[i] / inputs.size());
            }

            std::vector<std::vector<double>> normalized_inputs(inputs.size(), std::vector<double>(inputs[0].size()));
            for (size_t i = 0; i < inputs.size(); ++i) {
                for (size_t j = 0; j < inputs[i].size(); ++j) {
                    normalized_inputs[i][j] = (inputs[i][j] - mean[j]) / std[j];
                }
            }

            weights.resize(normalized_inputs[0].size());
            std::random_device rd;
            std::mt19937 gen(rd());
            std::normal_distribution<double> distribution(0.0, 1.0);
            for (size_t i = 0; i < weights.size(); ++i) {
                weights[i] = distribution(gen) * std::sqrt(1.0 / normalized_inputs[0].size());
            }

            for (int epoch = 0; epoch < epochs; ++epoch) {
                for (size_t i = 0; i < normalized_inputs.size(); ++i) {
                    double prediction = 0.0;
                    for (size_t j = 0; j < normalized_inputs[i].size(); ++j) {
                        prediction += normalized_inputs[i][j] * weights[j];
                    }
                    double error = is_classified ? categorical_crossentropy(labels[i], prediction) : mse(labels[i], prediction);
                    for (size_t j = 0; j < weights.size(); ++j) {
                        weights[j] += learning_rate * normalized_inputs[i][j] * error;
                    }
                }
            }
            std::cout << "training finished" << std::endl;
        }

        std::vector<double> predict(const std::vector<double>& input) {
            std::vector<double> result(input.size());
            double dot_product = 0.0;
            for (size_t i = 0; i < input.size(); ++i) {
                dot_product += input[i] * weights[i];
            }
            dot_product += bias;

            if (is_classified) {
                result = sigmoid({dot_product});
                return {result[0] >= 0.5 ? 1 : -1};
            } else {
                return {dot_product};
            }
        }

        void save_data(const std::string& filename) {
            std::ofstream file(filename, std::ios::binary);
            if (file.is_open()) {
                size_t weight_size = weights.size();
                file.write(reinterpret_cast<char*>(&weight_size), sizeof(size_t));
                file.write(reinterpret_cast<char*>(weights.data()), weights.size() * sizeof(double));
                file.write(reinterpret_cast<char*>(&bias), sizeof(double));
                file.write(reinterpret_cast<char*>(&learning_rate), sizeof(double));
                file.write(reinterpret_cast<char*>(&epochs), sizeof(int));
                file.write(reinterpret_cast<char*>(&is_classified), sizeof(bool));
                file.close();
            }
        }

        void load_data(const std::string& filename) {
            std::ifstream file(filename, std::ios::binary);
            if (file.is_open()) {
                size_t weight_size;
                file.read(reinterpret_cast<char*>(&weight_size), sizeof(size_t));
                weights.resize(weight_size);
                file.read(reinterpret_cast<char*>(weights.data()), weights.size() * sizeof(double));
                file.read(reinterpret_cast<char*>(&bias), sizeof(double));
                file.read(reinterpret_cast<char*>(&learning_rate), sizeof(double));
                file.read(reinterpret_cast<char*>(&epochs), sizeof(int));
                file.read(reinterpret_cast<char*>(&is_classified), sizeof(bool));
                file.close();
            }
        }
    };
}