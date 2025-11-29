import numpy as np
import pickle
import os

class QLearningAgent:
    def __init__(
        self,
        action_space,
        alpha=0.3,
        gamma=0.99,
        epsilon=0.3,
        epsilon_decay=0.9995,
        epsilon_min=0.01,
        alpha_decay=1.0,
        alpha_min=0.05,
        grid_size=10,
    ):
        """
        Basic tabular Q-Learning agent with simple vision-based state extraction.

        Parameters:
        - action_space: gymnasium action space
        - alpha: learning rate
        - gamma: discount factor
        - epsilon: initial exploration rate
        - epsilon_decay: multiplicative epsilon decay per episode/step (caller decides when to call decay)
        - epsilon_min: exploration floor
        - alpha_decay: multiplicative alpha decay factor applied in decay_alpha()
        - alpha_min: minimal learning rate
        - grid_size: bin size (in pixels) used to discretize positions
        """
        self.action_space = action_space
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.alpha_decay = alpha_decay
        self.alpha_min = alpha_min
        self.grid_size = grid_size
        self.q_table = {}
        self.prev_ball_x = None
        self.prev_ball_y = None

    def get_positions(self, observation):
        """
        Extracts raw positions from observation.
        Returns: (ball_x, ball_y, paddle_x)
        """
        if len(observation.shape) != 3:
            return None, None, None

        h, w, _ = observation.shape

        # Detect Paddle
        bottom_h = 30
        bottom = observation[-bottom_h:, :, :]
        red = bottom[:, :, 0]
        green = bottom[:, :, 1]
        blue = bottom[:, :, 2]
        mask_paddle = (red > 150) & (green < 100) & (blue < 100)

        if mask_paddle.any():
            ys_p, xs_p = np.where(mask_paddle)
            paddle_x = int(np.median(xs_p))
        else:
            paddle_x = w // 2 # Default center

        # Detect Ball
        top_cut = 30
        bottom_cut = 10 # Ignore very bottom
        play_area = observation[top_cut:h-bottom_cut, :, :]
        
        red = play_area[:, :, 0]
        green = play_area[:, :, 1]
        blue = play_area[:, :, 2]
        
        ball_pixels = (red > 150) & (green < 140) & (blue < 140)

        if ball_pixels.any():
            ys, xs = np.where(ball_pixels)
            idx = np.argmax(ys) # Lowest pixel
            ball_x = int(xs[idx])
            ball_y = int(ys[idx]) + top_cut
        else:
            ball_x = None
            ball_y = None
            
        return ball_x, ball_y, paddle_x

    def get_state(self, observation):
        """
        Extracts state from observation.
        State: (ball_x_bin, ball_y_bin, paddle_x_bin, ball_dx, ball_dy)
        """
        ball_x, ball_y, paddle_x = self.get_positions(observation)

        # Discretize
        if ball_x is None:
            # Return a generic "no ball" state.
            return (-1, -1, -1, 0, 0)

        # Bins
        # Screen is roughly 160x210
        grid_size = self.grid_size
        # Clamp to avoid out-of-range values if detection is noisy
        ball_x_bin = int(np.clip(ball_x // grid_size, 0, 160 // grid_size))
        ball_y_bin = int(np.clip(ball_y // grid_size, 0, 210 // grid_size))
        paddle_x_bin = int(np.clip(paddle_x // grid_size, 0, 160 // grid_size))

        # Velocity
        if self.prev_ball_x is None or self.prev_ball_y is None:
            dx = 0
            dy = 0
        else:
            dx = 1 if ball_x > self.prev_ball_x else (-1 if ball_x < self.prev_ball_x else 0)
            dy = 1 if ball_y > self.prev_ball_y else (-1 if ball_y < self.prev_ball_y else 0)
        
        self.prev_ball_x = ball_x
        self.prev_ball_y = ball_y

        return (ball_x_bin, ball_y_bin, paddle_x_bin, dx, dy)

    def get_action(self, observation, training=True):
        state = self.get_state(observation)
        
        if state is None:
            return self.action_space.sample()

        if training and np.random.random() < self.epsilon:
            return self.action_space.sample()
        
        # Greedy action
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.action_space.n)

        q_values = self.q_table[state]
        # Random tie-breaking among best actions (prevents action bias)
        max_q = np.max(q_values)
        best_actions = np.flatnonzero(q_values == max_q)
        return int(np.random.choice(best_actions))

    def update(self, observation, action, reward, next_observation, done=False):
        state = self.get_state(observation)
        next_state = self.get_state(next_observation)

        if state is None:
            return

        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.action_space.n)
        
        # Compute TD target (no bootstrap if terminal)
        if done:
            target = reward
        else:
            if next_state not in self.q_table:
                self.q_table[next_state] = np.zeros(self.action_space.n)
            target = reward + self.gamma * np.max(self.q_table[next_state])
        
        self.q_table[state][action] += self.alpha * (target - self.q_table[state][action])

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def decay_alpha(self):
        self.alpha = max(self.alpha_min, self.alpha * self.alpha_decay)

    def reset_episode(self):
        """Call at the beginning of each episode to clear internal momentum."""
        self.prev_ball_x = None
        self.prev_ball_y = None

    def save(self, filename):
        # Backward-compatible: keep saving only the q_table
        with open(filename, 'wb') as f:
            pickle.dump(self.q_table, f)
        print(f"Q-table saved to {filename}")

    def load(self, filename):
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                self.q_table = pickle.load(f)
            print(f"Q-table loaded from {filename}")
        else:
            print(f"No Q-table found at {filename}, starting fresh.")
