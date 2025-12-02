# DQN en Stable-Baselines3

SB3 ofrece una implementación de DQN basada en la versión clásica de Deep Q-Learning. DQN en SB3 usa las técnicas fundamentales de Q-learning con redes neuronales: memoria de repetición (replay buffer), red objetivo (target network) y clipping de gradiente para estabilizar el entrenamiento. 


Importante: la implementación de SB3 es la versión “vanilla” de DQN — no incluye extensiones como Double-DQN, Dueling-DQN ni Prioritized Experience Replay. 


Las políticas disponibles para DQN en SB3 son:
- MlpPolicy: adecuada para entornos con observaciones vectoriales/tabulares. 

- CnnPolicy: adecuada para entornos con input visual (imágenes). 

- MultiInputPolicy: para entornos con observaciones tipo diccionario/combinadas.

## Cuándo se puede usar DQN en SB3
- Acciones discretas: la acción debe pertenecer a un espacio Discrete. 
- Observaciones continuas, discretas o visuales: el espacio de observación puede ser Box, MultiDiscrete, MultiBinary o estructuras compatibles.
- No soporta políticas recurrentes.

## Parámetros importantes de la clase DQN

Al inicializar un agente DQN en SB3, se pueden ajustar múltiples hiperparámetros. Algunos de los más relevantes son:

| Parámetro                                                                  | Propósito / Efecto                                                                                                                                                                        |
| -------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `learning_rate`                                                            | Tasa de aprendizaje (puede ser constante o schedulada)                                                                                        |
| `buffer_size`                                                              | Capacidad del replay buffer (cantidad máxima de transiciones almacenadas)                                                                    |
| `learning_starts`                                                          | Número de pasos de recopilación antes de comenzar el entrenamiento — fase de “warm-up”.                                                          |
| `batch_size`                                                               | Tamaño de los minibatches usados para cada actualización (gradient step).                                                                      |
| `train_freq`                                                               | Cada cuántos pasos del entorno se realiza un entrenamiento (update).                                                                             |
| `gradient_steps`                                                           | Cuántos pasos de gradiente ejecutar luego de cada `train_freq`.                                                                                  |
| `gamma`                                                                    | Factor de descuento (discount factor) para valores futuros.                                                                                       |
| `target_update_interval` / `tau`                                           | Controlan el momento y la forma de actualizar la red objetivo (target network), ya sea mediante “hard update” periódica o “soft update” (Polyak). |
| `exploration_initial_eps`, `exploration_final_eps`, `exploration_fraction` | Parametrizan la estrategia ϵ-greedy de exploración, desde un ϵ inicial alto hasta un ϵ final más bajo.                                  |


## Ventajas y limitaciones de DQN en SB3

### Ventajas
- Implementación madura, sencilla de usar, con soporte para entornos tabulares y visuales.
- Maneja automáticamente replay buffer, target network, logueo, guardado y carga de modelos.
- Buen punto de partida para experimentar con entornos discretos (como juegos Atari, entornos clásicos de Gym, etc.).

### Limitaciones
- No soporta políticas recurrentes.
- Para problemas complejos o muy ruidosos, puede requerir mucho tuning de hiperparámetros.

