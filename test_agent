import gym
import numpy as np
from dqn_agent import DQNAgent  # pastikan file dqn_agent.py ada di folder yang sama

# --- Inisialisasi environment ---
env = gym.make("CartPole-v1", render_mode="human")  # tampilkan window saat running
state_size = env.observation_space.shape[0]  # ukuran state (4 untuk CartPole)
action_size = env.action_space.n             # jumlah aksi (2 untuk CartPole)

# --- Inisialisasi agent ---
agent = DQNAgent(state_size, action_size)
agent.epsilon = 0.01  # minimal eksplorasi saat testing

# --- Loop episode testing ---
for e in range(5):  # coba 5 episode
    # Reset environment
    state, _ = env.reset()                # Gym v0.26+ mengembalikan (obs, info)
    state = np.reshape(state, [1, state_size])

    # Loop langkah dalam episode
    for time in range(500):  # maksimal 500 langkah per episode
        env.render()
        
        # pilih aksi dari agent
        action = agent.act(state)

        # step di environment (Gym v0.26+ return 5 nilai)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # ubah bentuk next_state jadi [1, state_size]
        next_state = np.reshape(next_state, [1, state_size])

        # update state
        state = next_state

        if done:
            print(f"Episode {e+1} selesai setelah {time+1} langkah")
            break

env.close()
