from itertools import repeat

import gymnasium as gym
import ale_py
import keyboard
import time
import matplotlib.pyplot as plt

# Crear entorno
env = gym.make(
    "ALE/Breakout-v5",
    render_mode="human",
    full_action_space=False,
    repeat_action_probability=0.1,
    obs_type="rgb"
)

obs, info = env.reset()

print("Acciones:", env.action_space.n)
print("Significado:", env.unwrapped.get_action_meanings())

# Mapeo de teclas a acciones
actionDict = {
    "w": 0,   # NOOP
    "s": 1,   # FIRE
    "d": 2,   # RIGHT
    "a": 3    # LEFT
}

totalReward = 0

while True:
    if keyboard.is_pressed("q"):   # salir
        break

    # Acción por defecto
    action = 0   # NOOP

    # Detectar teclas que está apretando el usuario
    for key, mapped_action in actionDict.items():
        if keyboard.is_pressed(key):
            action = mapped_action

    # Ejecutar acción
    obs, reward, terminated, truncated, info = env.step(action)
    totalReward += reward

    if terminated or truncated:
        print(f"Juego terminado. Recompensa total: {totalReward}")
        totalReward = 0
        obs, info = env.reset()

    time.sleep(0.03)  # FPS ≈ 30

env.close()
