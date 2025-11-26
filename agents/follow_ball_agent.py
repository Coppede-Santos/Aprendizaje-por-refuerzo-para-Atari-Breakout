import gymnasium as gym
import ale_py
import time
import numpy as np

class FollowBallAgent:
    def __init__(self, action_space):
        self.action_space = action_space
        self.waiting_for_fire = True       # si es True, la próxima acción es FIRE
        self.frames_without_ball = 0       # cuántos frames llevamos sin ver la pelota
        self.prev_ball_x = None
    def reset(self):
        """Reset the agent for a new episode."""
        self.waiting_for_fire = True
        self.frames_without_ball = 0

    def detect_ball(self, observation):
        """
        Detecta la X de la pelota:
        - Roja
        - Zona media de juego
        - Siempre por encima de la barra (paddle_y - margin)
        """
        if len(observation.shape) != 3:
            self.frames_without_ball += 1
            return None

        h, w, _ = observation.shape

        # 1) Estimamos la Y de la barra (similar a detect_paddle)
        bottom_h = 20
        bottom = observation[-bottom_h:, :, :]
        red = bottom[:, :, 0]
        green = bottom[:, :, 1]
        blue = bottom[:, :, 2]
        mask_paddle = (red > 150) & (green < 140) & (blue < 140)

        if mask_paddle.any():
            ys_p, xs_p = np.where(mask_paddle)
            y_med_p = int(np.median(ys_p))
            paddle_y = h - bottom_h + y_med_p
        else:
            # fallback: barra "ideal" cerca del fondo
            paddle_y = h - 10

        # 2) Definimos zona de búsqueda de pelota:
        top_cut = 40
        margin = 4  # cuántos píxeles por encima de la barra cortamos

        y_max = int(paddle_y - margin)
        if y_max <= top_cut + 1:
            # No hay espacio útil para buscar pelota
            self.frames_without_ball += 1
            return None

        play_area = observation[top_cut:y_max, :, :]

        red = play_area[:, :, 0]
        green = play_area[:, :, 1]
        blue = play_area[:, :, 2]

        ball_pixels = (red > 150) & (green < 140) & (blue < 140)

        if ball_pixels.any():
            ys, xs = np.where(ball_pixels)
            # píxel rojo más abajo dentro de esta zona (pero ya nunca es la barra)
            idx = np.argmax(ys)
            x_local = xs[idx]

            ball_x = int(x_local)  # ya está en coords globales en X
            if ball_x == 8:
                self.frames_without_ball += 1
            else:
                self.frames_without_ball = 0
            return ball_x

        # si no encontramos nada:
        self.frames_without_ball += 1
        return None

    def detect_paddle(self, observation):
        """
        Detect the paddle's X coordinate.
        The paddle is typically at the bottom of the screen.
        """
        if len(observation.shape) == 3:
            # Focus on the bottom portion of the screen (where paddle is)
            bottom_region = observation[-30:, :, :]
            
            # Look for the paddle color (usually red/orange in Breakout)
            # Red channel is high, others are low
            red_pixels = (bottom_region[:, :, 0] > 150) & \
                        (bottom_region[:, :, 1] < 100) & \
                        (bottom_region[:, :, 2] < 100)
            
            if red_pixels.any():
                y_coords, x_coords = np.where(red_pixels)
                paddle_x = int(np.median(x_coords))
                return paddle_x
        
        return None

    def get_action(self, observation):
        """
        Decide action based on ball and paddle positions.
        """

        # Si estamos esperando lanzar la pelota, FIRE
        if self.waiting_for_fire:
            self.waiting_for_fire = False
            return 1  # FIRE

        ball_x = self.detect_ball(observation)
        paddle_x = self.detect_paddle(observation)


        # Heurística: si hace muchos frames que no vemos la pelota,
        # asumimos que hay una vida nueva esperando FIRE.
        if self.frames_without_ball > 15:
            self.waiting_for_fire = True
            self.prev_ball_x = None
            return 1  # FIRE

        # Si no vemos pelota o paleta → NOOP
        if ball_x is None or paddle_x is None:
            self.prev_ball_x = ball_x
            return 0

        # ---------------------
        # OFFSET según dirección
        # ---------------------
        offset = 5  # AJUSTÁ ENTRE 6 Y 15 SEGÚN RENDIMIENTO

        # Detectar movimiento en X
        if self.prev_ball_x is None:
            vx = 0
        else:
            vx = ball_x - self.prev_ball_x  # >0 derecha, <0 izquierda

        if vx > 0:  # pelota viniendo hacia la derecha
            target_x = ball_x - offset
        elif vx < 0:  # pelota viniendo hacia la izquierda
            target_x = ball_x + offset
        else:
            target_x = ball_x

        # Clamp por seguridad
        h, w, _ = observation.shape
        target_x = max(0, min(w - 1, int(target_x)))

        # Guardar ball_x para el próximo frame
        self.prev_ball_x = ball_x
        # ---------------------

        # Move paddle hacia target_x (en vez de hacia ball_x)
        threshold = 1

        if target_x < paddle_x - threshold:
            return 3  # LEFT
        elif target_x > paddle_x + threshold:
            return 2  # RIGHT
        else:
            return 0  # NOOP

def main():
    # Un print para estar seguros de que corre este main:
    print(">>> MAIN FOLLOW BALL BREAKOUT <<<")

    # Create environment
    env = gym.make(
        "ALE/Breakout-v5",
        render_mode="human",
        full_action_space=False,
        repeat_action_probability=0.1,
        obs_type="rgb"
    )

    agent = FollowBallAgent(env.action_space)

    print("Action Space:", env.action_space)
    print("Action Meanings:", env.unwrapped.get_action_meanings())

    num_episodes = 5

    for episode in range(num_episodes):
        obs, info = env.reset()
        agent.reset()  # Reset agent for new episode
        total_reward = 0
        steps = 0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            action = agent.get_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            time.sleep(0.01)  # Speed up slightly compared to human play

        print(f"Episode {episode + 1}: Total Reward = {total_reward}, Steps = {steps}")

    env.close()


if __name__ == "__main__":
    main()