<p style="text-align:center; font-size:20px;">
<strong>LCC - Facultad de Ingeniería - UNCUyo</strong><br>
<strong>Inteligencia Artificial 1</strong> &nbsp;&nbsp;&nbsp; Prof. Carlos A. Catania (Harpo)
</p>

---

<p style="text-align:center; font-size:24px;">
<em>Aprendizaje por refuerzo para Atari Breakout</em>
</p>

<p style="text-align:center;">
<strong>Código:</strong> ARABIM
</p>

<p style="text-align:center;">
<strong>Integrantes:</strong> Coppede Santos Ignacio, Sorbello Mauro
</p>

---

<p align="center">
  <img src="https://www.coolmathgames.com/sites/default/files/styles/mobile_game_image/public/Breakout_OG-logo.jpg.webp?itok=whyuODT0" 
       alt="Breakout" width="600">
</p>

## Introducción

**Atari Breakout**, lanzado el **13 de mayo de 1976**, es un videojuego arcade clásico.  
La dinámica del juego consiste en que el jugador controla una **raqueta horizontal** ubicada en la parte inferior de la pantalla, moviéndola de izquierda a derecha.

En la parte superior se encuentra una **banda de ladrillos**.  
Una **bola** desciende y el jugador debe golpearla con la raqueta, haciendo que rebote hacia arriba para impactar y destruir los ladrillos, los cuales desaparecen al ser golpeados. El objetivo es **eliminar por completo la pared de ladrillos**.

El presente proyecto busca **entrenar un agente mediante aprendizaje por refuerzo**, evaluando mediante métricas específicas su capacidad para:

## Marco Teórico

## Diseño Experimental

### Métricas Principales
Las siguientes métricas fueron utilizadas para la evaluación comparativa del desempeño de cada agente.

#### 1. Recompensa Total (Total Reward)
- **Definición**: La suma acumulada de las recompensas obtenidas en un episodio.
- **Fuente**: Retornado por `env.step()` como `reward`.
- **Interpretación**: Indica qué tan bien jugó el agente. Mayor puntaje es mejor.

#### 2. Duración del Episodio (Episode Length)
- **Definición**: Cantidad total de acciones ejecutadas (`steps`) hasta que el episodio termina.
- **Fuente**: Contador interno incrementado en cada iteración del bucle principal.
- **Interpretación**: Episodios más largos sugieren que el agente pudo mantener la pelota en juego durante más tiempo, aunque no necesariamente implica buen puntaje si no destruyó ladrillos.

#### 3. Golpes a Ladrillos (Brick Hits)
- **Definición**: Número de veces que la bola golpea un ladrillo.
- **Fuente**: Inferido cuando `reward > 0`.
- **Interpretación**: Mide la efectividad ofensiva del agente. Valores altos indican que el agente logró avanzar en la destrucción de la pared de ladrillos.


## Herramientas y Entornos

Para el desarrollo del proyecto se utilizaron diferentes librerías y herramientas dentro del ecosistema de Python, empleando la versión **3.13.9** del lenguaje.

Con respecto al entorno de ejecución, se utilizó **ALE-py** junto a **Gymnasium**, que proporcionan una interfaz estándar para la ejecución de juegos de Atari. En particular, se trabajó con la versión **"Breakout-v5"**, y para los agentes basados en aprendizaje profundo se empleó también la variante **"BreakoutNoFrameskip-v4"**, necesaria para el correcto funcionamiento de algoritmos de control como DQN.

El script de evaluación crea dos tipos de entornos según el agente:
- Para **Random**, **FollowBall** y **QLearning**, se utilizó:
  ```python
  gym.make(
      "ALE/Breakout-v5",
      obs_type="rgb",
      full_action_space=False,
      repeat_action_probability=0.1
  )
Este entorno proporciona observaciones RGB y un espacio de acciones reducido, facilitando el procesamiento y la comparación entre agentes no profundos.

- Para el agente DQN (Stable-Baselines3), se utilizó un entorno vectorizado:
    ```python
    seed = 42 
    env = make_atari_env("BreakoutNoFrameskip-v4", n_envs=1, seed=seed)
    env = VecFrameStack(env, n_stack=4)
  
Esta configuración es estándar en experimentos con Deep RL, ya que permite apilar frames consecutivos para capturar información temporal que el modelo necesita.

A diferencia de los agentes clásicos, que se ejecutaron localmente, el agente DQN fue entrenado en la plataforma Kaggle, aprovechando su disponibilidad de GPU gratuita y compatibilidad con Stable-Baselines3.
El modelo final fue exportado como: best_model.zip

## Estrategia de entrenamiento

### DQN

Para entrenar el agente DQN se realizaron diversas pruebas preliminares con el fin de establecer una configuración estable y eficiente dentro de las limitaciones de la plataforma Kaggle. Estas pruebas permitieron ajustar tanto los hiperparámetros como la estructura del entorno de entrenamiento hasta obtener los resultados finales utilizados para las 10 millones de iteraciones.

El objetivo principal fue maximizar la cantidad de ladrillos destruidos y la supervivencia del agente, priorizando comportamientos que aumentaran la recompensa total y evitaran pérdidas tempranas de vidas. Para ello se utilizó el esquema estándar de recompensas del entorno Breakout, sin modificaciones adicionales, dejando que el agente aprendiera las dinámicas óptimas únicamente a partir de los puntos otorgados por el juego original.

Con respecto al diseño del entorno, se empleó el preprocesamiento clásico para Atari: convertir las observaciones en escala de grises, reescalarlas a 84×84 píxeles y apilar de a 4 frames consecutivos. Este procedimiento permite que la red convolucional capture información temporal, como velocidad y dirección de la pelota.

Los hiperparámetros definitivos utilizados durante el entrenamiento fueron:

#### Hiperparámetros:

- **learning_rate**: 1e-4  
- **learning_starts**: 50 000 pasos  
- **exploration_fraction**: 0.10  
- **exploration_final_eps**: 0.01  
- **buffer_size**: 250 000 transiciones  
- **batch_size**: 32  
- **train_freq**: 4  
- **target_update_interval**: 10 000  
- **n_envs**: 8 entornos paralelos  
- **steps_training**: 10 millones de pasos

Estuvieron basados en las recomendaciones brindadas por la documentación.

Además, para garantizar estabilidad durante el entrenamiento, se utilizaron mecanismos de evaluación periódica y checkpoints automáticos.  
- Cada **50 000** pasos se evaluó el agente en 10 episodios.  
- Cada **1 millón** de pasos se guardó un checkpoint del modelo.  
- Al finalizar se guardó el modelo definitivo `dqn_breakout_FINAL_10M.zip`.

En conjunto, esta estrategia permitió entrenar un agente DQN robusto, capaz de aprender comportamientos complejos en Breakout, aprovechando procesamiento paralelo y GPU gratuita de Kaggle.