import random
import time
import numpy as np
import cv2 as cv

# Environment

start_time = time.time()

array = [[-1, -1, -1, -50, -1],
         [-1, -1, -50, 100, -1],
         [-1, -1, -50, -50, -1],
         [-1, -50, -1, -1, -1],
         [-1, -1, -1, -50, -50],
         [-1, -1, -1, -1, -1]]
array = np.array(array)

rows, cols = array.shape
cell_size = 50
actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
directions = ["Up", "Down", "Left", "Right"]

# Create canvas
image = np.zeros((rows * cell_size, cols * cell_size, 3), dtype=np.uint8)

# Color map: BGR format
color_map = {
    -50: (0, 0, 255),     # Red
    -1: (255, 255, 200),  # Light blue
    100: (0, 255, 0),     # Green
}

# Q-table
Q = np.zeros((rows, cols, len(actions)))

# Parameters
alpha = 0.1        # learning rate
gamma = 0.9        # discount factor
epsilon = 0.7      # exploration rate
episodes = 10000
start = (5, 0)
goal = (1, 3)
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

# Tiles
for i in range(rows):
    for j in range(cols):
        value = array[i][j]
        color = color_map.get(value, (50, 50, 50))
        top_left = (j * cell_size, i * cell_size)
        bottom_right = ((j + 1) * cell_size, (i + 1) * cell_size)
        cv.rectangle(image, top_left, bottom_right, color, thickness=-1)

# Grid Lines
for i in range(rows + 1):
    cv.line(image, (0, i * cell_size), (cols * cell_size, i * cell_size), (100, 100, 100), 1)
for j in range(cols + 1):
    cv.line(image, (j * cell_size, 0), (j * cell_size, rows * cell_size), (100, 100, 100), 1)

# Path
for (i, j) in path:
    center = (j * cell_size + cell_size // 2, i * cell_size + cell_size // 2)
    cv.circle(image, center, 8, (0, 255, 255), -1)  # Yellow

# Mark start and goal
start_center = (start[1] * cell_size + cell_size // 2, start[0] * cell_size + cell_size // 2)
goal_center = (goal[1] * cell_size + cell_size // 2, goal[0] * cell_size + cell_size // 2)
cv.circle(image, start_center, 10, (255, 0, 0), -1)  # Blue start
cv.circle(image, goal_center, 10, (0, 255, 0), -1)   # Green goal

# Show the image using OpenCV
cv.imshow("RL Path", image)

end_time = time.time()
print("Runtime:", end_time - start_time, "seconds")

cv.waitKey(0)
cv.destroyAllWindows()
