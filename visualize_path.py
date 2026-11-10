
# Import Matplotlib for plotting the grid and UAV route.
import matplotlib.pyplot as plt

# Import the route planner.
from path_planner import find_path

# Import the independent safety validator.
from safety_validator import validate_path


# Define the mission configuration.
start = (0, 0)
goal = (9, 7)
obstacles = [(2, 0), (2, 1), (2, 2)]
grid_size = 10

# Generate a route using A*.
path = find_path(start, goal, obstacles, grid_size)

# Validate the route before visualizing it.
is_safe, message = validate_path(
    path=path,
    start=start,
    goal=goal,
    obstacles=obstacles,
    grid_size=grid_size,
)

# Stop if planning failed or the route did not pass validation.
if path is None or not is_safe:
    raise RuntimeError(f"Route rejected: {message}")

# Separate the route coordinates for plotting.
x_coords = [point[0] for point in path]
y_coords = [point[1] for point in path]

# Create the figure and axes.
fig, ax = plt.subplots(figsize=(9, 8))

# Draw the route as a connected line with waypoint markers.
ax.plot(
    x_coords,
    y_coords,
    marker="o",
    linewidth=2,
    label="Validated UAV route",
)

# Draw every obstacle cell.
for x, y in obstacles:
    ax.scatter(
        x, y,
        marker="s",
        s=220,
        color="red",
        label="Obstacle" if (x, y) == obstacles[0] else None,
    )

# Mark the start position.
ax.scatter(
    start[0], start[1],
    marker="o",
    s=180,
    color="green",
    label="Start",
    zorder=5,
)

# Mark the goal position.
ax.scatter(
    goal[0], goal[1],
    marker="*",
    s=260,
    color="gold",
    edgecolors="black",
    label="Goal",
    zorder=5,
)

# Add grid lines at cell boundaries.
ax.set_xticks(range(grid_size))
ax.set_yticks(range(grid_size))
ax.set_xlim(-0.5, grid_size - 0.5)
ax.set_ylim(-0.5, grid_size - 0.5)
ax.grid(True, alpha=0.4)

# Keep cells square and label the axes.
ax.set_aspect("equal")
ax.set_xlabel("X coordinate")
ax.set_ylabel("Y coordinate")
ax.set_title(
    f"UAV A* Path Planning | Moves: {len(path) - 1}\n"
    f"Safety validation: {'PASS' if is_safe else 'FAIL'}"
)

# Display a legend and the finished plot.
ax.legend()
plt.tight_layout()
plt.show()
