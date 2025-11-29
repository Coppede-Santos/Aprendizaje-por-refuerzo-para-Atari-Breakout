import numpy as np
import random
from collections import deque
import os

import torch
import torch.nn as nn
import torch.optim as optim


class ConvQNetwork(nn.Module):
    """
    CNN para Atari Breakout
    Input: (batch, num_frames=4, H, W) -- H y W reducidos (p.ej. 105x80)
    """
    def __init__(self, num_frames, n_actions):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(num_frames, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        # LazyLinear se ajusta solo al tamaño que salga de la conv
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        # state, next_state: np.array (num_frames, H, W), dtype uint8
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        # np.array mantiene uint8, no lo cambiamos acá
        return np.array(states), actions, rewards, np.array(next_states), dones

    def __len__(self):
        return len(self.buffer)


class ConvDQNAgent:
    def __init__(
        self,
        action_space,
        gamma=0.99,
        lr=1e-4,
        epsilon_start=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.999995,   # decae por paso
        buffer_size=30_000,       # MENOS para no volar la RAM
        batch_size=32,
        target_update_freq=10_000,
        learn_start=10_000,       # empezar a entrenar después de 10k transiciones
        num_frames=4,
        device=None,
    ):
        self.action_space = action_space
        self.n_actions = action_space.n
        self.gamma = gamma

        # Epsilon-greedy
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.learn_start = learn_start

        self.total_steps = 0
        self.learn_steps = 0

        self.num_frames = num_frames
        self.frames = deque(maxlen=self.num_frames)  # para frame stacking

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Redes
        self.q_net = ConvQNetwork(self.num_frames, self.n_actions).to(self.device)
        self.target_net = ConvQNetwork(self.num_frames, self.n_actions).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)

    # =========================
    # PROCESADO Y DOWNSAMPLE DE UN FRAME
    # =========================
    def _process_single_obs(self, obs):
        """
        obs: (H, W, 3) uint8 (RGB)
        -> frame_small: (H2, W2) uint8 en [0,255]
        Donde H2 = H/2, W2 = W/2 (con promedio 2x2)
        """
        obs = np.array(obs, copy=False)
        if obs.ndim == 3 and obs.shape[2] == 3:
            # RGB -> gris
            r = obs[:, :, 0].astype(np.float32)
            g = obs[:, :, 1].astype(np.float32)
            b = obs[:, :, 2].astype(np.float32)
            gray = 0.299 * r + 0.587 * g + 0.114 * b
        else:
            gray = obs.astype(np.float32)

        # Downsample 2x2 por promedio: (H,W) -> (H/2, W/2)
        h, w = gray.shape
        h2 = h // 2
        w2 = w // 2
        gray = gray[: 2 * h2, : 2 * w2]  # asegurar divisibilidad
        gray_small = gray.reshape(h2, 2, w2, 2).mean(axis=(1, 3))  # (h2,w2)

        # Volver a 0-255 uint8
        gray_small = np.clip(gray_small, 0, 255).astype(np.uint8)
        return gray_small  # (H2,W2) uint8

    def _get_state_array(self):
        """
        Devuelve el estado actual: np.array (num_frames, H2, W2), dtype uint8
        """
        assert len(self.frames) == self.num_frames
        return np.stack(self.frames, axis=0)

    def reset_episode(self, first_obs):
        """
        Llamar al inicio de cada episodio con la primera observación del env.
        Rellena el stack con el mismo frame reducido.
        """
        frame = self._process_single_obs(first_obs)
        self.frames.clear()
        for _ in range(self.num_frames):
            self.frames.append(frame)

    # =========================
    # INTERACCIÓN CON EL ENTORNO
    # =========================
    def get_action(self, training=True):
        """
        Usa el stack interno de frames para elegir acción.
        """
        self.total_steps += 1

        # Decaimos epsilon por paso
        if training:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # Explorar
        if training and np.random.rand() < self.epsilon:
            return self.action_space.sample()

        # Explotar
        state = self._get_state_array()  # (C,H2,W2), uint8
        state_tensor = torch.from_numpy(state).unsqueeze(0).float().to(self.device)  # (1,C,H2,W2)
        state_tensor /= 255.0
        with torch.no_grad():
            q_values = self.q_net(state_tensor)
            action = torch.argmax(q_values, dim=1).item()
        return int(action)

    def step(self, next_obs, action, reward, done):
        """
        Maneja una transición:
        - Genera (state, next_state)
        - Mete en el replay buffer
        - Entrena un paso
        - Actualiza el frame stack interno
        """
        # Estado actual
        state = self._get_state_array()  # (C,H2,W2), uint8

        # Procesar nuevo frame
        next_frame = self._process_single_obs(next_obs)
        next_frames = deque(self.frames, maxlen=self.num_frames)
        next_frames.append(next_frame)
        next_state = np.stack(next_frames, axis=0).astype(np.uint8)

        # Guardar en buffer
        self.replay_buffer.push(state, action, reward, next_state, done)

        # Actualizar frames internos
        self.frames = next_frames

        # Entrenar
        self.train_step()

    # =========================
    # ENTRENAMIENTO DQN
    # =========================
    def train_step(self):
        if len(self.replay_buffer) < max(self.batch_size, self.learn_start):
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # states / next_states: uint8 -> float32 en [0,1]
        states = torch.from_numpy(states).float().to(self.device) / 255.0
        next_states = torch.from_numpy(next_states).float().to(self.device) / 255.0
        actions = torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        # Q(s,a)
        q_values = self.q_net(states).gather(1, actions)

        # Q_target(s',a')
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(dim=1, keepdim=True)[0]
            target = rewards + self.gamma * (1 - dones) * next_q_values

        loss = nn.MSELoss()(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.learn_steps += 1

        if self.learn_steps % self.target_update_freq == 0:
            self.update_target_net()

    def update_target_net(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

    # =========================
    # GUARDAR / CARGAR
    # =========================
    def save(self, filename):
        checkpoint = {
            "q_net_state_dict": self.q_net.state_dict(),
            "target_net_state_dict": self.target_net.state_dict(),
            "epsilon": self.epsilon,
            "total_steps": self.total_steps,
            "learn_steps": self.learn_steps,
        }
        torch.save(checkpoint, filename)
        print(f"[ConvDQN] Modelo guardado en {filename}")

    def load(self, filename):
        if os.path.exists(filename):
            checkpoint = torch.load(filename, map_location=self.device)
            self.q_net.load_state_dict(checkpoint["q_net_state_dict"])
            self.target_net.load_state_dict(checkpoint["target_net_state_dict"])
            self.epsilon = checkpoint.get("epsilon", self.epsilon)
            self.total_steps = checkpoint.get("total_steps", 0)
            self.learn_steps = checkpoint.get("learn_steps", 0)
            print(f"[ConvDQN] Modelo cargado desde {filename}")
        else:
            print(f"[ConvDQN] No se encontró {filename}, empezando desde cero.")
