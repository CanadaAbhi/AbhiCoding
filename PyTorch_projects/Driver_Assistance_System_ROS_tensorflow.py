#The ADAS system aims to:
#Simulate crash scenarios in a custom environment.
#Train a reinforcement learning (RL) agent to avoid crashes.
#Use ROS for real-world integration (e.g., sensor data and control).

# bash
#pip install tensorflow tensorflow-agents matplotlib gym
#pip install rospy rospkg
# Environment Definition
#Define a custom driving environment using OpenAI Gym for crash scenario simulation.


import gym
from gym import spaces
import numpy as np

class DrivingEnv(gym.Env):
    def __init__(self):
        super(DrivingEnv, self).__init__()
        # Define action space: [accelerate, brake, steer]
        self.action_space = spaces.Box(low=np.array([-1, 0, -1]), high=np.array([1, 1, 1]), dtype=np.float32)

        # Define observation space: [vehicle speed, distance to obstacle, steering angle]
        self.observation_space = spaces.Box(low=0, high=100, shape=(3,), dtype=np.float32)

        # Initialize state
        self.state = np.array([50.0, 20.0, 0.0])  # Example: speed=50, distance=20m, steering=0

    def reset(self):
        self.state = np.array([50.0, 20.0, 0.0])
        return self.state

    def step(self, action):
        # Update state based on action
        speed, distance, steering = self.state
        accel, brake, steer = action

        # Update speed and distance
        speed = max(0, speed + accel * 5 - brake * 10)
        distance -= speed * 0.1
        steering += steer * 10

        # Check for crash or success
        done = False
        reward = -1  # Default penalty
        if distance <= 0:
            done = True
            reward = -100  # Crash penalty
        elif distance > 50:
            done = True
            reward = 100  # Success reward

        self.state = np.array([speed, max(0, distance), steering])
        return self.state, reward, done, {}

#Reinforcement Learning with TF-Agents

import tensorflow as tf
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment

# Convert Gym environment to TF-Agents environment
gym_env = DrivingEnv()
train_env = tf_py_environment.TFPyEnvironment(suite_gym.wrap_env(gym_env))
eval_env = tf_py_environment.TFPyEnvironment(suite_gym.wrap_env(gym_env))
#Define RL Agent
#Use the DQN (Deep Q-Network) agent for training.


from tf_agents.agents.dqn import dqn_agent
from tf_agents.networks.q_network import QNetwork
from tf_agents.utils import common

# Define Q-network architecture
q_net = QNetwork(train_env.observation_spec(), train_env.action_spec())

# Define optimizer and agent
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
agent = dqn_agent.DqnAgent(
    train_env.time_step_spec(),
    train_env.action_spec(),
    q_network=q_net,
    optimizer=optimizer,
    td_errors_loss_fn=common.element_wise_squared_loss,
)
agent.initialize()


from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.drivers.dynamic_step_driver import DynamicStepDriver

# Replay buffer for storing experiences
replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=train_env.batch_size,
    max_length=100000,
)

# Driver for collecting trajectories
from tf_agents.policies.random_tf_policy import RandomTFPolicy

random_policy = RandomTFPolicy(train_env.time_step_spec(), train_env.action_spec())
driver = DynamicStepDriver(
    train_env,
    random_policy,
    observers=[replay_buffer.add_batch],
    num_steps=200,
)

# Populate replay buffer with initial data
driver.run(train_env.reset())



from tf_agents.metrics import tf_metrics

# Dataset from replay buffer
dataset = replay_buffer.as_dataset(
    num_parallel_calls=3,
    sample_batch_size=64,
    num_steps=2,
).prefetch(3)

iterator = iter(dataset)

# Training loop
num_iterations = 10000

for iteration in range(num_iterations):
    # Sample a batch of experiences from the replay buffer
    experience, _ = next(iterator)
    
    # Train the agent on the batch of experiences
    loss_info = agent.train(experience)
    
    if iteration % 100 == 0:
        print(f"Iteration {iteration}, Loss: {loss_info.loss.numpy()}")


#Integrate TensorFlow with ROS to control a simulated or real vehicle.
#Create a ROS node that uses the trained model for predictions.


import rospy
from std_msgs.msg import Float32MultiArray

class ROSDriverNode:
    def __init__(self):
        rospy.init_node("driver_assistance")
        
        # Publisher for actions (accelerate/brake/steer)
        self.action_pub = rospy.Publisher("/vehicle/actions", Float32MultiArray, queue_size=10)
        
        # Subscriber for vehicle state (speed/distance/steering)
        rospy.Subscriber("/vehicle/state", Float32MultiArray, self.state_callback)
        
        self.agent_policy = agent.policy
    
    def state_callback(self, msg):
        state = tf.convert_to_tensor([msg.data], dtype=tf.float32)
        
        # Predict action using RL policy
        action_step = self.agent_policy.action(state)
        
        # Publish action to vehicle controller
        action_msg = Float32MultiArray()
        action_msg.data = action_step.action.numpy()[0]
        self.action_pub.publish(action_msg)

if __name__ == "__main__":
    node = ROSDriverNode()
    rospy.spin()


num_episodes = 10
total_rewards = []

for _ in range(num_episodes):
    time_step = eval_env.reset()
    episode_reward = 0
    
    while not time_step.is_last():
        action_step = agent.policy.action(time_step)
        time_step = eval_env.step(action_step.action)
        
        episode_reward += time_step.reward.numpy()
    
    total_rewards.append(episode_reward)

print(f"Average Reward: {np.mean(total_rewards)}")