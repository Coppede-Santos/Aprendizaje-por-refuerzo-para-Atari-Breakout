# Algoritmos de Aprendizaje por Refuerzo: Q-learning y DQN

Este documento ofrece una visión general de dos algoritmos de aprendizaje por refuerzo relevantes para videojuegos tipo Atari: **Q-learning tabular** y **Deep Q-Network (DQN)**. Se explican sus fundamentos, mejoras y consideraciones para implementarlos en entornos como Atari Breakout.

---

## 1. Q-learning clásico (tabular)

### 1.1 Concepto básico  
- Es un algoritmo de *off-policy* para estimar la función de valor \( Q(s, a) \), es decir, la calidad de tomar acción \( a \) en estado \( s \).  
- Se usa una **tabla Q** (matriz) que almacena valores para cada combinación estado-acción.  
- La actualización se hace usando la ecuación de Bellman:

\[
Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]
\]

donde:  
- \( \alpha \) es la tasa de aprendizaje,  
- \( \gamma \) es el factor de descuento,  
- \( r \) es la recompensa recibida al ir a \( s' \).

### 1.2 Fortalezas y limitaciones  
**Fortalezas:**
- Es simple y fácil de entender.
- Muy eficaz cuando el espacio de estados es pequeño o puede discretizarse de forma razonable.

**Limitaciones:**
- No escala bien: si el espacio de estados es muy grande (por ejemplo, píxeles de un juego Atari), la tabla se vuelve inmanejable.  
- Requiere discretización si el estado es continuo o muy grande, lo que puede llevar a perder buena parte de la información importante del entorno.

---

## 2. Deep Q-Network (DQN)

Cuando el espacio de estados es muy grande (por ejemplo, frames de videojuegos), la aproximación tabular ya no es práctica. Aquí es donde entra **DQN**, un algoritmo que combina Q-learning con redes neuronales profundas.

### 2.1 Fundamentos de DQN  
- En lugar de una tabla Q, se usa una **red neuronal** que toma como entrada el estado (por ejemplo, una imagen del juego) y devuelve un vector con los valores \( Q(s, a) \) para cada acción posible. :contentReference[oaicite:0]{index=0}  
- Para entrenar esta red se minimiza el error de predicción entre el Q estimado por la red y un **target** calculado a partir de la ecuación de Bellman, de forma similar a Q-learning. :contentReference[oaicite:1]{index=1}  
- Se emplean varias técnicas clave para estabilizar el entrenamiento:

  1. **Replay Buffer (memoria de experiencia):** se guardan transiciones \((s, a, r, s')\) en un buffer y, durante el entrenamiento, se muestrean minibatches de forma aleatoria para actualizar la red. Esto rompe la correlación entre muestras consecutivas y mejora la eficiencia del uso de datos. :contentReference[oaicite:2]{index=2}  
  2. **Target Network (red objetivo):** se usa una copia de la red principal para calcular los valores objetivo \( Q(s', a) \). Esta red se actualiza (sincroniza) periódicamente con los pesos de la red principal, lo que ayuda a que los objetivos no cambien demasiado rápido y estabiliza el aprendizaje. :contentReference[oaicite:3]{index=3}  
  3. **Preprocesamiento de las observaciones:** para videojuegos tipo Atari, las imágenes suelen convertirse a escala de grises, reducirse de tamaño (por ejemplo a \(84 \times 84\) píxeles) y apilar varios frames para capturar el movimiento. :contentReference[oaicite:4]{index=4}  
  4. **Reward Clipping (opcional):** los rewards se pueden “clipear” a un rango fijo (por ejemplo \([-1, +1]\)) para evitar que valores extremos desestabilicen el entrenamiento. :contentReference[oaicite:5]{index=5}  

### 2.2 Variantes y mejoras  
- **Double DQN (DDQN):** corrige el sesgo por sobreestimación de valores \( Q \) (overestimation) al separar las redes para elegir la acción y para estimar su valor. :contentReference[oaicite:6]{index=6}  
- **Prioritized Experience Replay:** en lugar de muestrear uniformemente del buffer, se priorizan aquellas transiciones con mayor error de predicción (TD error), para aprender más rápido de las experiencias “importantes”. :contentReference[oaicite:7]{index=7}  
- Otras mejoras pueden incluir **Dueling DQN**, **Noisy Nets**, **Double + Dueling**, entre otras (depende de los recursos y la complejidad que quieras asumir).

### 2.3 Consideraciones prácticas para Atari  
- El entrenamiento con DQN para juegos Atari requiere muchos pasos de interacción (millones), ya que las redes necesitan suficiente experiencia para aprender. :contentReference[oaicite:8]{index=8}  
- Es recomendable usar GPU para acelerar el entrenamiento, porque las convoluciones sobre imágenes son costosas.  
- Elegir hiperparámetros adecuados: tamaño del buffer, tasa de aprendizaje, frecuencia de actualización de la target network, política de exploración (epsilon-greedy), tamaño del batch, etc. :contentReference[oaicite:9]{index=9}  
- Tener un mecanismo de evaluación para verificar que el agente realmente mejora: no solo entrenar, sino guardar checkpoints, episodios de evaluación y métricas.

---

## 3. Comparación entre Q-learning clásico y DQN

| Característica | Q-learning tabular | DQN (Deep Q-Network) |
|---|---|---|
| Representación de la función Q | Tabla explícita | Red neuronal |
| Escalabilidad con el espacio de estados | Baja (no escala bien) | Alta (se adapta a espacios grandes) |
| Memoria requerida | Depende del número de estados | Depende del tamaño de la red + buffer de experiencia |
| Estabilidad de aprendizaje | Puede ser bastante estable si los estados están bien definidos | Necesita técnicas como Replay Buffer y Target Network para ser estable |
| Uso típico | Entornos discretos pequeños o medianos | Entornos con observaciones complejas (imágenes, estados continuos) |

---

## 4. Conclusión y recomendación para el proyecto Breakout

Para un entorno como **Atari Breakout**, donde los estados son imágenes de la pantalla, la **mejor opción** práctica es usar **DQN**, no Q-learning tabular:

- El espacio de estados es demasiado grande para una tabla tradicional.  
- DQN permite usar una red convolucional para procesar las imágenes y estimar valores \( Q(s, a) \).  
- Las técnicas de replay buffer y target network son esenciales para que el entrenamiento sea estable.

Además, se puede considerar usar una variante mejorada (por ejemplo, Double DQN) si el entrenamiento básico no converge o si ves que el agente sobreestima valores de acción.

---

## 5. Referencias

- Mnih, V. et al. *Human-level control through deep reinforcement learning.* Nature, 2015. (paper original de DQN) :contentReference[oaicite:10]{index=10}  
- Step-by-step Double Deep Q-Networks tutorial, Yinxuan Li. :contentReference[oaicite:11]{index=11}  
- Paul Kroe, “DQN for Atari Breakout from Scratch” (blog con explicación de target network, replay buffer, etc.) :contentReference[oaicite:12]{index=12}  
- Explicación de DQN (Next Electronics) con formulación matemática. :contentReference[oaicite:13]{index=13}  
- Prioritized Experience Replay (Schaul et al.) :contentReference[oaicite:14]{index=14}  
- “DQN a partir de píxeles” con enfoque aplicado a Atari. :contentReference[oaicite:15]{index=15}  

