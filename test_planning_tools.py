
# Import both tool functions.
from planning_tools import plan_uav_path, validate_uav_path


# Define the test mission.
obstacles = [[2, 0], [2, 1], [2, 2]]

# Generate a route using the planning tool.
planning_result = plan_uav_path(
    start_x=0,
    start_y=0,
    goal_x=9,
    goal_y=7,
    obstacles=obstacles,
    grid_size=10
)

# Display the route returned by the tool.
print("Planning result:", planning_result)

# Validate the generated route using the separate safety tool.
validation_result = validate_uav_path(
    path=planning_result["path"],
    start_x=0,
    start_y=0,
    goal_x=9,
    goal_y=7,
    obstacles=obstacles,
    grid_size=10
)

# Display the independent validation result.
print("Validation result:", validation_result)

# Confirm that the test produced a route and the validator accepted it.
assert planning_result["success"] is True
assert validation_result["safe"] is True

# Report successful completion of the test.
print("Both planning and validation tools passed.")
