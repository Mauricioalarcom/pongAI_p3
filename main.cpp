#include "include/utec/algebra/tensor.h"
#include "include/utec/nn/neural_network.h"
#include "include/utec/nn/dense.h"
#include "include/utec/nn/activation.h"
#include "include/utec/agent/PongAgent.h"
#include <iostream>
#include <memory>

using namespace utec::algebra;
using namespace utec::neural_network;
using namespace utec::agent;

void demo_tensor_operations() {
    std::cout << "=== TENSOR ALGEBRA DEMO ===\n";

    // Create and manipulate tensors
    Tensor<float, 2> matrix(3, 3);
    matrix.fill(2.0f);
    matrix(1, 1) = 5.0f;

    std::cout << "Matrix operations:\n";
    std::cout << "matrix(1,1) = " << matrix(1, 1) << "\n";

    auto scaled = matrix * 2.0f;
    std::cout << "scaled(1,1) = " << scaled(1, 1) << "\n";

    // Test transpose
    auto transposed = matrix.transpose_2d();
    std::cout << "Transpose shape: [" << transposed.shape()[0] << ", " << transposed.shape()[1] << "]\n\n";
}

void demo_neural_network() {
    std::cout << "=== NEURAL NETWORK DEMO ===\n";

    // Create a simple neural network
    NeuralNetwork<float> net;
    net.add_layer(std::make_unique<Dense<float>>(2, 4));
    net.add_layer(std::make_unique<ReLU<float>>());
    net.add_layer(std::make_unique<Dense<float>>(4, 1));

    // EJEMPLO DE ENTRENAMIENTO SIMPLE
    std::cout << "Entrenando red neuronal...\n";

    // Datos XOR simples
    Tensor<float, 2> X(4, 2);
    X(0,0)=0; X(0,1)=0; X(1,0)=0; X(1,1)=1;
    X(2,0)=1; X(2,1)=0; X(3,0)=1; X(3,1)=1;

    Tensor<float, 2> Y(4, 1);
    Y(0,0)=0; Y(1,0)=1; Y(2,0)=1; Y(3,0)=0;

    // Entrenar con el método avanzado
    auto metrics = net.train_advanced(X, Y, 1000, 0.1f, 20, 1e-6f);

    std::cout << "Entrenamiento completado!\n";
    std::cout << "Épocas: " << metrics.epochs_trained << "\n";
    std::cout << "Loss final: " << metrics.final_loss << "\n";

    // Probar predicciones
    std::cout << "Probando predicciones:\n";
    auto predictions = net.forward(X);
    for (int i = 0; i < 4; ++i) {
        std::cout << "Input: [" << X(i,0) << "," << X(i,1)
                  << "] -> Esperado: " << Y(i,0)
                  << ", Predicho: " << predictions(i,0) << "\n";
    }

    // Create some dummy data for original demo
    Tensor<float, 2> input(1, 2);
    input(0, 0) = 0.5f;
    input(0, 1) = 0.3f;

    auto output = net.forward(input);
    std::cout << "Neural network output: " << output(0, 0) << "\n\n";
}

void demo_pong_agent() {
    std::cout << "=== PONG AGENT DEMO ===\n";

    // Create a neural network for the agent
    auto agent_net = std::make_unique<NeuralNetwork<float>>();
    agent_net->add_layer(std::make_unique<Dense<float>>(3, 8));
    agent_net->add_layer(std::make_unique<ReLU<float>>());
    agent_net->add_layer(std::make_unique<Dense<float>>(8, 3));

    PongAgent<float> agent(std::move(agent_net));
    EnvGym env;

    std::cout << "Running PONG simulation...\n";

    float total_reward = 0.0f;
    int episodes = 5;

    for (int episode = 0; episode < episodes; ++episode) {
        auto state = env.reset();
        bool done = false;
        float episode_reward = 0.0f;
        int steps = 0;

        std::cout << "\nEpisode " << (episode + 1) << ":\n";

        while (!done && steps < 100) { // Limit steps to prevent infinite loops
            int action = agent.act(state);
            float reward;
            state = env.step(action, reward, done);

            episode_reward += reward;
            steps++;

            if (steps <= 10 || steps % 20 == 0) { // Show first 10 steps and every 20th step
                std::cout << "  Step " << steps << ": action=" << action
                         << ", ball=(" << state.ball_x << "," << state.ball_y << ")"
                         << ", paddle=" << state.paddle_y
                         << ", reward=" << reward << "\n";
            }

            if (done) {
                std::cout << "  Episode ended after " << steps << " steps\n";
                break;
            }
        }

        total_reward += episode_reward;
        std::cout << "  Episode reward: " << episode_reward << "\n";
    }

    std::cout << "\nTotal reward across " << episodes << " episodes: " << total_reward << "\n";
    std::cout << "Average reward per episode: " << (total_reward / static_cast<float>(episodes)) << "\n\n";
}

void demo_training_example() {
    std::cout << "=== TRAINING DEMO (XOR) ===\n";

    // XOR training example
    Tensor<float, 2> X(4, 2);
    X(0,0)=0; X(0,1)=0;
    X(1,0)=0; X(1,1)=1;
    X(2,0)=1; X(2,1)=0;
    X(3,0)=1; X(3,1)=1;

    Tensor<float, 2> Y(4, 1);
    Y(0,0)=0; Y(1,0)=1; Y(2,0)=1; Y(3,0)=0;

    NeuralNetwork<float> net;
    net.add_layer(std::make_unique<Dense<float>>(2, 4));
    net.add_layer(std::make_unique<ReLU<float>>());
    net.add_layer(std::make_unique<Dense<float>>(4, 1));

    std::cout << "Training XOR function...\n";
    float final_loss = net.train(X, Y, 500, 0.1f);

    std::cout << "Final loss: " << final_loss << "\n";

    // Test predictions
    auto predictions = net.forward(X);
    std::cout << "XOR Results:\n";
    for (int i = 0; i < 4; ++i) {
        std::cout << "  [" << X(i,0) << "," << X(i,1) << "] -> "
                 << predictions(i,0) << " (target: " << Y(i,0) << ")\n";
    }
    std::cout << "\n";
}

int main() {
    std::cout << "PONG AI - Complete Demonstration\n";
    std::cout << "================================\n\n";

    try {
        demo_tensor_operations();
        demo_neural_network();
        demo_training_example();
        demo_pong_agent();

        std::cout << "=== ALL DEMOS COMPLETED SUCCESSFULLY ===\n";
        std::cout << "\nTo run individual tests:\n";
        std::cout << "  ./test_tensor\n";
        std::cout << "  ./test_neural_network\n";
        std::cout << "  ./test_agent_env\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
