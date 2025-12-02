from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
import ale_py
import gymnasium
import os

gymnasium.register_envs(ale_py)

# 1. Crear el mismo entorno
seed = 42 # Puedes cambiar este número
env = make_atari_env("BreakoutNoFrameskip-v4", n_envs=1, seed=seed)
env = VecFrameStack(env, n_stack=4)

# 2. Cargar tu archivo .zip (asegúrate de que esté en la misma carpeta)
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "mi_agente_breakout")
model = DQN.load(model_path, env=env)

# 3. Jugar
obs = env.reset()
while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = env.step(action)
    env.render("human") # Esto abrirá la ventanita del juego
