# --- trainDDPG.py ---

from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from hydraulic_env import HydraulicActuatorEnv
import numpy as np

# Create environment
env = DummyVecEnv([lambda: HydraulicActuatorEnv()])
env = VecNormalize(env, norm_obs=True, norm_reward=True)

# Define action noise
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=1.0 * np.ones(n_actions))

# Initialize model with better hyperparameters
model = DDPG(
    "MlpPolicy",
    env,
    action_noise=action_noise,
    learning_rate=5e-4,
    buffer_size=100_000,
    batch_size=128,
    tau=0.005,
    gamma=0.95,
    verbose=1,
    tensorboard_log="./DDPG_hydraulic_logs/"
)

# Train
print("Starting training...")
eval_callback = EvalCallback(env, eval_freq=2000, n_eval_episodes=5, deterministic=True, verbose=1)
model.learn(total_timesteps=300_000, callback=eval_callback)

# Evaluate
def evaluate(model, env, n_episodes=10, dt=1e-3):
    rewards = []
    v_log = []
    time_log = []
    env.training = False
    env.norm_reward = False

    for _ in range(n_episodes):
        obs = env.reset()
        done = False
        total_reward = 0.0
        step_count = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            v_log.append(action[0])
            time_log.append(step_count * dt)
            total_reward += reward
            step_count += 1
        rewards.append(total_reward)
    return np.mean(rewards), np.std(rewards)

mean, std = evaluate(model, env)
print(f"Avg reward: {mean:.2f} ± {std:.2f}")

# Save the model
model.save("ddpg_hydraulic_actuator")

