import gymnasium as gym
import ale_py
import time
import numpy as np

class FollowBallAgent:
    def __init__(self, action_space):
        self.action_space = action_space
        # Actions: 0=NOOP, 1=FIRE, 2=RIGHT, 3=LEFT
        self.first_action = True  # Track if this is the first action
    
    def reset(self):
        """Reset the agent for a new episode."""
        self.first_action = True
        
    def detect_ball(self, observation):
        """
        Detect the ball's X coordinate from the RGB observation.
        The ball is red in Breakout.
        """
        if len(observation.shape) == 3:
            # Exclude the bottom region (where paddle is) and top region (where red bricks are)
            # to avoid confusion since ball, paddle, and some bricks are red
            # Focus on the middle playing area
            middle_region = observation[60:-30, :, :]  # Skip top 60 pixels and bottom 30 pixels
            
            # Look for red pixels (the ball is red)
            # Red channel is high, green and blue are low
            ball_pixels = (middle_region[:, :, 0] > 150) & \
                         (middle_region[:, :, 1] < 100) & \
                         (middle_region[:, :, 2] < 100)
            
            if ball_pixels.any():
                # Find coordinates of ball pixels
                y_coords, x_coords = np.where(ball_pixels)
                
                # Use the median X coordinate of red pixels
                # (helps filter noise and focuses on the ball)
                ball_x = int(np.median(x_coords))
                return ball_x
        
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
        # First action of the episode should always be FIRE to launch the ball
        if self.first_action:
            self.first_action = False
            return 1  # FIRE
        
        ball_x = self.detect_ball(observation)
        paddle_x = self.detect_paddle(observation)
        
        # If we can't detect either, fire to start/continue game
        if ball_x is None or paddle_x is None:
            return 1  # FIRE
        
        # Move paddle towards ball
        threshold = 5  # Dead zone to avoid jittering
        
        if ball_x < paddle_x - threshold:
            return 3  # LEFT
        elif ball_x > paddle_x + threshold:
            return 2  # RIGHT
        else:
            return 0  # NOOP (aligned)

def main():
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