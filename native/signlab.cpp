// Implémentations personnelles C++17, interface C stable pour ctypes.
// kind : 0 = modèle linéaire softmax ; 1 = SVM hinge ; 2 = MLP ; 3 = RBF.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifdef _WIN32
#define API extern "C" __declspec(dllexport)
#else
#define API extern "C" __attribute__((visibility("default")))
#endif

namespace {
thread_local std::string last_error;
using Vec = std::vector<double>;
struct Model {
    int kind, d, k, h;
    double gamma;
    Vec W1, b1, W2, b2, centers;
    Model(int kind_, int d_, int k_, int h_, double gamma_, unsigned seed)
        : kind(kind_), d(d_), k(k_), h((kind_ >= 2) ? h_ : 0), gamma(gamma_) {
        if (kind < 0 || kind > 3 || d < 1 || d > 65536 || k < 1 || k > 1024 || h < 0 || h > 4096 || !std::isfinite(gamma) || gamma <= 0)
            throw std::invalid_argument("Dimensions ou paramètres natifs invalides");
        if (kind >= 2 && h < 1) throw std::invalid_argument("Une couche cachée est nécessaire");
        if (static_cast<long long>(d) * std::max(k, h) > 20000000) throw std::invalid_argument("Modèle natif trop volumineux");
        std::mt19937 rng(seed);
        std::normal_distribution<double> normal(0, 1);
        if (kind == 2) {
            W1.resize(d * h); b1.assign(h, 0);
            for (double &v : W1) v = normal(rng) * std::sqrt(2.0 / (d + h));
        }
        if (kind == 3) centers.resize(d * h);
        int width = kind >= 2 ? h : d;
        W2.assign(width * k, 0); b2.assign(k, 0);
        if (kind == 2) for (double &v : W2) v = normal(rng) * std::sqrt(2.0 / (h + k));
    }
    Vec basis(const double *x) const {
        if (kind < 2) return Vec(x, x + d);
        Vec z(h, 0);
        for (int j = 0; j < h; ++j) {
            if (kind == 2) {
                double sum = b1[j];
                for (int f = 0; f < d; ++f) sum += x[f] * W1[f * h + j];
                z[j] = std::tanh(sum);
            } else {
                double distance = 0;
                for (int f = 0; f < d; ++f) { double delta = x[f] - centers[j * d + f]; distance += delta * delta; }
                z[j] = std::exp(-gamma * distance);
            }
        }
        return z;
    }
    Vec scores(const Vec &z) const {
        Vec result = b2;
        for (std::size_t f = 0; f < z.size(); ++f)
            for (int c = 0; c < k; ++c) result[c] += z[f] * W2[f * k + c];
        return result;
    }
    void fit(const double *X, const int *y, int n, int epochs, double lr, double l2, unsigned seed) {
        if (!X || !y || n < 2 || epochs < 1 || epochs > 100000 || !std::isfinite(lr) || lr <= 0 || !std::isfinite(l2) || l2 < 0)
            throw std::invalid_argument("Données ou budget natifs invalides");
        for (long long i = 0; i < static_cast<long long>(n) * d; ++i)
            if (!std::isfinite(X[i])) throw std::invalid_argument("Entrée native non finie");
        for (int i = 0; i < n; ++i) if (y[i] < 0 || y[i] >= k) throw std::invalid_argument("Label natif hors limites");
        std::mt19937 rng(seed);
        if (kind == 3) {
            if (h > n) throw std::invalid_argument("Plus de centres que d'observations");
            std::vector<int> order(n); std::iota(order.begin(), order.end(), 0); std::shuffle(order.begin(), order.end(), rng);
            for (int j = 0; j < h; ++j) std::copy(X + order[j] * d, X + (order[j] + 1) * d, centers.begin() + j * d);
            for (int iteration = 0; iteration < 40; ++iteration) {
                Vec sums(h * d, 0); std::vector<int> counts(h, 0);
                for (int i = 0; i < n; ++i) {
                    int best = 0; double distance_best = std::numeric_limits<double>::infinity();
                    for (int j = 0; j < h; ++j) {
                        double dist = 0;
                        for (int f = 0; f < d; ++f) { double delta = X[i * d + f] - centers[j * d + f]; dist += delta * delta; }
                        if (dist < distance_best) { best = j; distance_best = dist; }
                    }
                    ++counts[best];
                    for (int f = 0; f < d; ++f) sums[best * d + f] += X[i * d + f];
                }
                double change = 0;
                for (int j = 0; j < h; ++j) if (counts[j]) for (int f = 0; f < d; ++f) {
                    double value = sums[j * d + f] / counts[j]; change += std::abs(value - centers[j * d + f]); centers[j * d + f] = value;
                }
                if (change < 1e-8) break;
            }
        }
        int width = kind >= 2 ? h : d;
        for (int epoch = 0; epoch < epochs; ++epoch) {
            Vec g1(W1.size(), 0), gb1(b1.size(), 0), g2(W2.size(), 0), gb2(k, 0);
            for (int i = 0; i < n; ++i) {
                Vec z = basis(X + i * d), s = scores(z), residual(k, 0);
                if (kind == 1) {
                    int rival = -1;
                    double margin = 0;
                    for (int c = 0; c < k; ++c) {
                        double candidate = s[c] - s[y[i]] + 1;
                        if (c != y[i] && candidate > margin) { rival = c; margin = candidate; }
                    }
                    if (rival >= 0) { residual[rival] = 1; residual[y[i]] = -1; }
                } else {
                    double maximum = *std::max_element(s.begin(), s.end()), sum = 0;
                    for (int c = 0; c < k; ++c) { residual[c] = std::exp(s[c] - maximum); sum += residual[c]; }
                    for (double &value : residual) value /= sum;
                    residual[y[i]] -= 1;
                }
                for (int c = 0; c < k; ++c) {
                    gb2[c] += residual[c] / n;
                    for (int f = 0; f < width; ++f) g2[f * k + c] += z[f] * residual[c] / n;
                }
                if (kind == 2) for (int j = 0; j < h; ++j) {
                    double delta = 0;
                    for (int c = 0; c < k; ++c) delta += residual[c] * W2[j * k + c];
                    delta *= (1 - z[j] * z[j]) / n;
                    gb1[j] += delta;
                    for (int f = 0; f < d; ++f) g1[f * h + j] += X[i * d + f] * delta;
                }
            }
            double rate = kind == 1 ? lr / (1 + 0.01 * epoch) : lr;
            for (std::size_t j = 0; j < W2.size(); ++j) W2[j] -= rate * (g2[j] + l2 * W2[j]);
            for (int j = 0; j < k; ++j) b2[j] -= rate * gb2[j];
            for (std::size_t j = 0; j < W1.size(); ++j) W1[j] -= rate * (g1[j] + l2 * W1[j]);
            for (std::size_t j = 0; j < b1.size(); ++j) b1[j] -= rate * gb1[j];
        }
    }
    Vec pack() const {
        Vec values{1.0, static_cast<double>(kind), static_cast<double>(d), static_cast<double>(k), static_cast<double>(h), gamma};
        for (const Vec *part : {&W1, &b1, &W2, &b2, &centers}) values.insert(values.end(), part->begin(), part->end());
        return values;
    }
};
}

API const char *sl_error() { return last_error.c_str(); }
API void *sl_create(int kind, int d, int k, int h, double gamma, unsigned seed) {
    try { return new Model(kind, d, k, h, gamma, seed); } catch (const std::exception &e) { last_error = e.what(); return nullptr; }
}
API void sl_free(void *handle) { delete static_cast<Model *>(handle); }
API int sl_fit(void *handle, const double *X, const int *y, int n, int epochs, double lr, double l2, unsigned seed) {
    try { if (!handle) throw std::invalid_argument("Handle nul"); static_cast<Model *>(handle)->fit(X, y, n, epochs, lr, l2, seed); return 0; }
    catch (const std::exception &e) { last_error = e.what(); return -1; }
}
API int sl_predict(void *handle, const double *X, int n, double *output) {
    try {
        if (!handle || !X || !output || n < 1) throw std::invalid_argument("Prédiction native invalide");
        const auto &model = *static_cast<Model *>(handle);
        for (int i = 0; i < n; ++i) {
            for (int f = 0; f < model.d; ++f) if (!std::isfinite(X[i * model.d + f])) throw std::invalid_argument("Entrée non finie");
            Vec scores = model.scores(model.basis(X + i * model.d));
            std::copy(scores.begin(), scores.end(), output + i * model.k);
        }
        return 0;
    } catch (const std::exception &e) { last_error = e.what(); return -1; }
}
API int sl_state_size(void *handle) { return handle ? static_cast<int>(static_cast<Model *>(handle)->pack().size()) : -1; }
API int sl_export(void *handle, double *output, int size) {
    try {
        if (!handle || !output) throw std::invalid_argument("Export natif invalide");
        Vec values = static_cast<Model *>(handle)->pack();
        if (size != static_cast<int>(values.size())) throw std::invalid_argument("Taille export invalide");
        std::copy(values.begin(), values.end(), output); return 0;
    } catch (const std::exception &e) { last_error = e.what(); return -1; }
}
API void *sl_import(const double *values, int size) {
    try {
        if (!values || size < 6 || values[0] != 1) throw std::invalid_argument("État natif invalide");
        for (int i = 0; i < size; ++i) if (!std::isfinite(values[i])) throw std::invalid_argument("État non fini");
        for (int i = 1; i <= 4; ++i) if (values[i] != std::floor(values[i]) || values[i] < 0 || values[i] > 65536) throw std::invalid_argument("Entier natif invalide");
        Model model(static_cast<int>(values[1]), static_cast<int>(values[2]), static_cast<int>(values[3]), static_cast<int>(values[4]), values[5], 0);
        if (size != static_cast<int>(model.pack().size())) throw std::invalid_argument("Taille état natif invalide");
        int offset = 6;
        for (Vec *part : {&model.W1, &model.b1, &model.W2, &model.b2, &model.centers}) {
            std::copy(values + offset, values + offset + part->size(), part->begin()); offset += static_cast<int>(part->size());
        }
        return new Model(std::move(model));
    } catch (const std::exception &e) { last_error = e.what(); return nullptr; }
}
