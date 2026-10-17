# evaluate_ddpg.py

from stable_baselines3 import DDPG
from hydraulic_env import HydraulicActuatorEnv
import matplotlib.pyplot as plt

# Load environment and model
env = HydraulicActuatorEnv()
model = DDPG.load("ddpg_hydraulic_actuator")

obs, _ = env.reset()
done = False

# Logs
y_log = []
target_log = []
time_log = []
v_log = []

# Run one episode
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)

    y_log.append(obs[0])
    target_log.append(env.target)  # Track target over time
    time_log.append(env.time)
    v_log.append(action[0])
# Plot response vs target
plt.figure(figsize=(8, 5))
plt.subplot(2, 1, 1)
#plt.figure(figsize=(8, 5))
plt.plot(time_log, y_log, label='System Response (y)', color='blue')
plt.plot(time_log, target_log, label='Target Position', linestyle='--', color='red')
plt.title("Target vs Response")
plt.xlabel("Time (s)")
plt.ylabel("Position (m)")
plt.legend()

plt.subplot(2, 1, 2)
#plt.figure(figsize=(8, 5))
plt.plot(time_log, v_log, label='Control Signal (V)', color='blue')
plt.title("Control Signal")
plt.xlabel("Time (s)")
plt.ylabel("Control (V)")
plt.legend()

plt.grid(True)
plt.tight_layout()
plt.show()

