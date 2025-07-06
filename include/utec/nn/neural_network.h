#pragma once

#include "layer.h"
#include "loss.h"
#include "utec/algebra/tensor.h"
#include <vector>
#include <memory>
#include <iostream>

namespace utec::neural_network {

template<typename T>
class NeuralNetwork : public ILayer<T> {
private:
    std::vector<std::unique_ptr<ILayer<T>>> layers;
    MSELoss<T> criterion;

public:
    void add_layer(std::unique_ptr<ILayer<T>> layer) {
        layers.push_back(std::move(layer));
    }

    // Ejecuta forward por todas las capas
    utec::algebra::Tensor<T,2> forward(const utec::algebra::Tensor<T,2>& x) override {
        if (layers.empty()) {
            throw std::runtime_error("No layers in network");
        }

        auto current = x;
        for (auto& layer : layers) {
            current = layer->forward(current);
        }
        return current;
    }

    // Lanza backward desde la última capa
    utec::algebra::Tensor<T,2> backward(const utec::algebra::Tensor<T,2>& grad) override {
        if (layers.empty()) {
            return grad;
        }

        auto current_grad = grad;
        // Backward pass through layers in reverse order
        for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
            current_grad = (*it)->backward(current_grad);
        }
        return current_grad;
    }

    // Actualiza todos los parámetros con learning rate lr
    void update_parameters(T lr) override {
        for (auto& layer : layers) {
            layer->update_parameters(lr);
        }
    }

    // Método de conveniencia para optimización
    void optimize(T lr) {
        update_parameters(lr);
    }

    // Entrena con X, Y durante epochs
    T train(const utec::algebra::Tensor<T,2>& X, const utec::algebra::Tensor<T,2>& Y, size_t epochs, T lr) {
        T final_loss = T(0);

        for (size_t epoch = 0; epoch < epochs; ++epoch) {
            // Forward pass
            auto predictions = forward(X);

            // Compute loss
            T loss = criterion.forward(predictions, Y);
            final_loss = loss;

            // Backward pass
            auto loss_grad = criterion.backward();
            backward(loss_grad);

            // Update parameters
            optimize(lr);

            // Optional: print loss every 100 epochs for debugging
            if (epoch % 100 == 0) {
                std::cout << "Epoch " << epoch << ", Loss: " << loss << std::endl;
            }
        }

        return final_loss;
    }

    // Get number of layers
    size_t size() const {
        return layers.size();
    }

    // Check if network is empty
    bool empty() const {
        return layers.empty();
    }
};

} // namespace utec::neural_network
