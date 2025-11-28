import numpy as np
import pickle
import os

class QLearningAgent:
    def __init__(self, action_space, alpha=0.5, gamma=0.99, epsilon=1.0, epsilon_decay=0.9995, epsilon_min=0.01):
        self.action_space = action_space
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
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
        # Grid size: 10x10 pixels?
        grid_size = 10
        ball_x_bin = ball_x // grid_size
        ball_y_bin = ball_y // grid_size
        paddle_x_bin = paddle_x // grid_size

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
        
        return int(np.argmax(self.q_table[state]))

    def update(self, observation, action, reward, next_observation):
        state = self.get_state(observation)
        next_state = self.get_state(next_observation)

        if state is None:
            return

        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.action_space.n)
        
        if next_state is None:
            # Terminal state or lost ball, assume 0 future value?
            # Or just ignore next max q
            target = reward
        else:
            if next_state not in self.q_table:
                self.q_table[next_state] = np.zeros(self.action_space.n)
            target = reward + self.gamma * np.max(self.q_table[next_state])
        
        self.q_table[state][action] += self.alpha * (target - self.q_table[state][action])

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, filename):
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
