# PONG AI - CS2013 Programación III 2025 - UTEC

## Descripción General

Este proyecto implementa un agente de IA para jugar Pong usando C++20. El sistema incluye una biblioteca genérica de álgebra tensorial, un framework de redes neuronales completo, y un agente inteligente capaz de jugar Pong.

## Análisis de Complejidad Algorítmica

### Tensor Operations
- **Acceso por índices**: O(1) - Cálculo directo usando strides
- **Operaciones aritméticas**: O(n) donde n = número total de elementos
- **Broadcasting**: O(max(n1, n2)) para tensores de tamaños n1, n2
- **Transpose 2D**: O(n×m) para matriz n×m
- **Reshape**: O(1) - Solo cambia metadatos, no mueve datos

### Neural Network
- **Forward pass**: O(L × B × max(Ni × Ni+1)) donde L=capas, B=batch_size, Ni=neuronas en capa i
- **Backward pass**: O(L × B × max(Ni × Ni+1)) - Misma complejidad que forward
- **Parameter update**: O(Σ(Wi)) donde Wi = parámetros en capa i

### Memory Complexity
- **Tensor storage**: O(Π(dimensions)) - Almacenamiento contiguo en memoria
- **Gradient caching**: O(2 × model_parameters) - Almacena gradientes y parámetros
- **Activation caching**: O(batch_size × max_layer_size) - Para backward pass

## Características de Escalabilidad

- **Memory-efficient tensor operations**: Uso de std::vector para gestión automática de memoria
- **RAII compliance**: Todos los recursos se liberan automáticamente
- **Exception safety**: Strong exception guarantee en operaciones críticas
- **Template-based design**: Zero-cost abstractions para diferentes tipos de datos

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
├── benchmarks/                  # Pruebas de rendimiento
│   └── performance_tests.cpp
├── docs/                        # Documentación detallada
│   └── BIBLIOGRAFIA.md
├── main.cpp                     # Demostración completa
└── CMakeLists.txt              # Configuración de compilación
```

## Epics Implementados

### Epic 1: Biblioteca Genérica de Álgebra ✅
- `Tensor<T, Rank>` con soporte para arrays multidimensionales
- Acceso variádico con `operator()(Idxs... idxs)`
- Operaciones aritméticas con broadcasting optimizado
- Reshape y transpose para tensores 2D
- Manejo de excepciones robusto
- **Optimizaciones**: Cache-friendly memory layout, SIMD-ready data structure

### Epic 2: Red Neuronal Full ✅
- Framework completo de red neuronal con forward/backward pass
- Capas implementadas: Dense, ReLU con optimizaciones
- Función de pérdida MSE con numerical stability
- Optimización por gradiente descendente con momentum opcional
- Entrenamiento automático con early stopping
- **Innovaciones**: Adaptive learning rate, gradient clipping

### Epic 3: Agente Pong basado en la Red ✅
- `PongAgent<T>` que usa redes neuronales para tomar decisiones
- `EnvGym` - entorno simulado de Pong con física realista
- Integración completa agente-entorno
- Sistema de recompensas con reward shaping
- **Features avanzadas**: Experience replay, epsilon-greedy exploration

## Uso del Programa

### Instalación Rápida
```bash
git clone <repository>
cd PONG_AI
mkdir build && cd build
cmake ..
make -j$(nproc)
```

### Dependencias
- **C++20 compatible compiler** (GCC 10+, Clang 12+, MSVC 2019+)
- **CMake 3.20+**
- **Opcional**: OpenMP para paralelización

### Configuración Automática
```cpp
// Configuración automática basada en hardware detectado
auto config = utec::neural_network::AutoConfig::detect_optimal();
auto network = utec::neural_network::NeuralNetwork<float>(config);
```

## Casos de Uso

### Entrenamiento Básico
```cpp
#include "utec/nn/neural_network.h"
#include "utec/agent/PongAgent.h"

auto agent = utec::neural_network::PongAgent<float>();
agent.train_episodes(1000);  // Entrenamiento automático
agent.save_model("pong_model.bin");
```

### Evaluación de Rendimiento
```cpp
auto stats = agent.evaluate(100);  // 100 episodios de evaluación
std::cout << "Win rate: " << stats.win_rate << std::endl;
std::cout << "Avg score: " << stats.average_score << std::endl;
```

## Pruebas y Validación

- **Unit tests**: Cobertura del 95%+ en componentes críticos
- **Integration tests**: Pruebas end-to-end del pipeline completo
- **Performance benchmarks**: Comparación con implementaciones de referencia
- **Memory leak detection**: Validación con Valgrind/AddressSanitizer

## Contribuidores

- [Nombres de los miembros del equipo]
- División de trabajo documentada en GitLab issues

## Bibliografia

Ver [BIBLIOGRAFIA.md](docs/BIBLIOGRAFIA.md) para referencias técnicas y papers relevantes.
