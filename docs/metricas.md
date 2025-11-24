# Métricas del Proyecto

Este documento define las métricas que se utilizarán para evaluar el desempeño de los agentes en el entorno de Atari Breakout.

## Métricas Principales

### 1. Recompensa Total (Total Reward)
- **Definición**: La suma acumulada de las recompensas obtenidas en un episodio.
- **Fuente**: Retornado por `env.step()` como `reward`.
- **Interpretación**: Indica qué tan bien jugó el agente. Mayor puntaje es mejor.

### 2. Vidas (Lives)
- **Definición**: El número de vidas restantes del agente.
- **Fuente**: `info['lives']`.
- **Interpretación**: Útil para penalizar la pérdida de vidas o para terminar el entrenamiento prematuramente si se desea. Comienza en 5.

### 3. Duración del Episodio (Episode Length)
- **Definición**: El número de pasos (frames) que duró el episodio.
- **Fuente**: Contador de pasos en el bucle principal o `info['episode_frame_number']`.
- **Interpretación**: Episodios más largos pueden indicar supervivencia, pero no necesariamente éxito si no se obtienen puntos.

### 4. Golpes a Ladrillos (Brick Hits)
- **Definición**: Número de veces que la bola golpea un ladrillo.
- **Fuente**: Inferido cuando `reward > 0`.
- **Interpretación**: Mide la efectividad ofensiva del agente.

## Métricas Derivadas (Opcionales)

### 5. Recompensa Promedio (Average Reward)
- Promedio de la recompensa total sobre los últimos N episodios (ventana móvil).
- Útil para suavizar la varianza y ver la tendencia de aprendizaje.

### 6. Máxima Recompensa (Max Reward)
- La recompensa más alta obtenida en un solo episodio durante el entrenamiento/evaluación.

## Implementación Técnica

El diccionario `info` retornado por Gymnasium (`ALE/Breakout-v5`) contiene:
```python
{
    'lives': 5,                 # Vidas restantes
    'episode_frame_number': 4,  # Frames en el episodio actual
    'frame_number': 4           # Frames totales desde el inicio
}
```

Para detectar un golpe a un ladrillo:
```python
if reward > 0:
    brick_hits += 1
```
