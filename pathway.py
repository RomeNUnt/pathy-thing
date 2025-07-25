import random
import time
import numpy as np

# Environment

start_time = time.time()

array = [[-1, -1, -1, -50, -1],
         [-1, -1, -50, 100, -1],
         [-1, -1, -50, -50, -1],
         [-1, -50, -1, -1, -1],
         [-1, -1, -1, -50, -1],
         [-1, -1, -1, -1, -1]]

rows, cols = len(array), len(array[0])
actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
directions = ["Up", "Down", "Left", "Right"]

# Q-table
Q = np.zeros((rows, cols, len(actions)))

# Parameters
alpha = 0.1        # learning rate
gamma = 0.9        # discount factor
epsilon = 0.7      # exploration rate
episodes = 1000
start = (5, 0)
max_steps = 100

# Training loop
for ep in range(episodes):
    i, j = start
    cost = 60
    for step in range(max_steps):
        state = (i, j)

        # Choose action (exploration vs exploitation)
        if random.uniform(0, 1) < epsilon:
            action_index = random.randint(0, 3)
        else:
            action_index = np.argmax(Q[i, j])

        di, dj = actions[action_index]
        ni, nj = i + di, j + dj

        # Check bounds
        if not (0 <= ni < rows and 0 <= nj < cols):
            continue

        reward = array[ni][nj]

        # Q-learning update
        old_q = Q[i, j, action_index]
        next_max = np.max(Q[ni, nj])
        new_q = old_q + alpha * (reward + gamma * next_max - old_q)
        Q[i, j, action_index] = new_q

        i, j = ni, nj
        cost += reward

        if reward == 100 or cost < 0:
            break

# After training, follow the best path
print("\nLearned path:")
i, j = start
cost = 60
path = [(i, j)]
direction_of_path = ["Start"]

while True:
    if array[i][j] == 100:
        print(f"Reached goal at {i, j} with cost {cost}")
        break
    if cost < 0:
        print("Failed: cost dropped below zero")
        break

    action_index = np.argmax(Q[i, j])
    di, dj = actions[action_index]
    ni, nj = i + di, j + dj

    if not (0 <= ni < rows and 0 <= nj < cols):
        print("Hit wall")
        break

    i, j = ni, nj
    path.append((i, j))
    direction_of_path.append(directions[action_index])
    cost += array[i][j]

print("Path:", direction_of_path)

end_time = time.time()
print("Runtime:", end_time - start_time, "seconds")
