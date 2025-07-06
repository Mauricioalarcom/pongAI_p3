# PONG AI - CS2013 Programación III 2025 - UTEC

## Descripción General

Este proyecto implementa un agente de IA para jugar Pong usando C++20. El sistema incluye una biblioteca genérica de álgebra tensorial, un framework de redes neuronales completo, y un agente inteligente capaz de jugar Pong.

## Estructura del Proyecto

```
pong_ai/
├── include/utec/
│   ├── algebra/tensor.h          # Biblioteca de álgebra tensorial
│   ├── nn/                       # Framework de red neuronal
│   │   ├── layer.h              # Interfaz base de capas
│   │   ├── dense.h              # Capa densa (fully connected)
│   │   ├── activation.h         # Funciones de activación (ReLU)
│   │   ├── loss.h               # Funciones de pérdida (MSE)
│   │   └── neural_network.h     # Red neuronal principal
│   └── agent/
│       └── PongAgent.h          # Agente y entorno de Pong
├── src/utec/agent/
│   └── PongAgent.cpp            # Implementación del agente
├── tests/                       # Casos de prueba
│   ├── test_tensor.cpp
│   ├── test_neural_network.cpp
│   └── test_agent_env.cpp
├── main.cpp                     # Demostración completa
└── CMakeLists.txt              # Configuración de compilación
```

## Epics Implementados

### Epic 1: Biblioteca Genérica de Álgebra ✅
- `Tensor<T, Rank>` con soporte para arrays multidimensionales
- Acceso variádico con `operator()(Idxs... idxs)`
- Operaciones aritméticas con broadcasting
- Reshape y transpose para tensores 2D
- Todos los casos de prueba especificados implementados

### Epic 2: Red Neuronal Full ✅
- Framework completo de red neuronal con forward/backward pass
- Capas implementadas: Dense, ReLU
- Función de pérdida MSE
- Optimización por gradiente descendente
- Entrenamiento automático con XOR como ejemplo

### Epic 3: Agente Pong basado en la Red ✅
- `PongAgent<T>` que usa redes neuronales para tomar decisiones
- `EnvGym` - entorno simulado de Pong con física básica
- Integración completa agente-entorno
- Sistema de recompensas implementado

## Características Principales

### Tensor Library
- **Flexibilidad**: Soporte para cualquier tipo numérico y rango
- **Performance**: Almacenamiento contiguo en memoria
- **Seguridad**: Verificación de límites en tiempo de ejecución
- **Broadcasting**: Multiplicación con broadcasting para dimensiones de tamaño 1

### Neural Network Framework
- **Modular**: Arquitectura basada en capas intercambiables
- **Extensible**: Interfaz `ILayer<T>` para nuevas capas
- **Eficiente**: Forward y backward pass optimizados
- **Completo**: Incluye optimización automática de parámetros

### Pong Agent
- **Inteligente**: Usa redes neuronales para tomar decisiones
- **Adaptable**: Puede usar cualquier arquitectura de red
- **Realista**: Física de juego simplificada pero funcional

## Compilación y Uso

### Requisitos
- C++20 compatible compiler (GCC 10+, Clang 12+, MSVC 2019+)
- CMake 3.20+

### Compilación
```bash
mkdir build
cd build
cmake ..
make
```

### Ejecución
```bash
# Demostración completa
./PONG_AI

# Tests individuales
./test_tensor
./test_neural_network
./test_agent_env
```

## Ejemplos de Uso

### Tensor Operations
```cpp
#include "utec/algebra/tensor.h"
using namespace utec::algebra;

// Crear tensor 2D
Tensor<float, 2> matrix(3, 3);
matrix.fill(2.0f);
matrix(1, 1) = 5.0f;

// Operaciones
auto scaled = matrix * 2.0f;
auto transposed = matrix.transpose_2d();
```

### Neural Network
```cpp
#include "utec/nn/neural_network.h"
using namespace utec::neural_network;

// Crear red
NeuralNetwork<float> net;
net.add_layer(std::make_unique<Dense<float>>(2, 4));
net.add_layer(std::make_unique<ReLU<float>>());
net.add_layer(std::make_unique<Dense<float>>(4, 1));

// Entrenar
float loss = net.train(X, Y, epochs, learning_rate);
```

### Pong Agent
```cpp
#include "utec/agent/PongAgent.h"
using namespace utec::agent;

// Crear agente
auto net = std::make_unique<NeuralNetwork<float>>();
// ... configurar red ...
PongAgent<float> agent(std::move(net));

// Jugar
EnvGym env;
auto state = env.reset();
int action = agent.act(state);
```

## Casos de Prueba Verificados

### Epic 1 - Tensor
- ✅ Creación, acceso y fill
- ✅ Reshape válido e inválido
- ✅ Suma y resta de tensores
- ✅ Multiplicación escalar y broadcasting
- ✅ Transpose 2D

### Epic 2 - Neural Network
- ✅ ReLU forward/backward
- ✅ MSE Loss forward/backward
- ✅ Dense layer operations
- ✅ XOR training convergence
- ✅ Shape mismatch detection

### Epic 3 - Pong Agent
- ✅ Instanciación básica
- ✅ Simulación de pasos
- ✅ Integración agente-entorno
- ✅ Condiciones límite
- ✅ Física del entorno

## Rendimiento y Optimizaciones

- **Memoria**: Almacenamiento contiguo para mejor cache locality
- **Cálculo**: Operaciones vectorizadas donde es posible
- **Gradientes**: Caching inteligente para backward pass
- **Broadcasting**: Implementación eficiente para tensores 2D

## Extensiones Futuras (Epic 4 & 5)

- **Paralelismo**: ThreadPool para procesamiento concurrente
- **CUDA**: Soporte para GPU acceleration
- **Algoritmos**: Implementación de DQN, SARSA
- **Serialización**: Guardar/cargar modelos entrenados
- **Métricas**: Sistema completo de evaluación

## Notas de Implementación

### Decisiones de Diseño
1. **Header-only**: Máxima compatibilidad y facilidad de uso
2. **Template-based**: Flexibilidad de tipos numéricos
3. **RAII**: Gestión automática de memoria
4. **Exception safety**: Manejo robusto de errores

### Limitaciones Conocidas
1. Broadcasting solo implementado completamente para tensores 2D
2. Optimizaciones específicas de CPU no implementadas
3. Algoritmos de entrenamiento avanzados pendientes

## Autores

Proyecto desarrollado para CS2013 - Programación III 2025 - UTEC

## Licencia

Este proyecto es parte del material académico de UTEC.
# pongAI_p3
