
# Define a function that independently validates a planned UAV route.
def validate_path(
    path,
    start,
    goal,
    obstacles,
    grid_size=10
):
    # Convert coordinates and obstacles into tuples for consistent checks.
    start = tuple(start)
    goal = tuple(goal)
    obstacles = {tuple(cell) for cell in obstacles}

    # Reject a missing or empty path.
    if not path:
        return False, "No path was provided."

    # Convert every waypoint into a tuple.
    path = [tuple(point) for point in path]

    # Check that the route starts at the required position.
    if path[0] != start:
        return False, "The path does not start at the required position."

    # Check that the route ends at the required destination.
    if path[-1] != goal:
        return False, "The path does not end at the required goal."

    # Check every waypoint individually.
    for index, (x, y) in enumerate(path):

        # Reject coordinates outside the grid.
        if not (0 <= x < grid_size and 0 <= y < grid_size):
            return False, f"Waypoint {index} is outside the grid: {(x, y)}."

        # Reject waypoints that intersect obstacles.
        if (x, y) in obstacles:
            return False, f"Waypoint {index} intersects an obstacle: {(x, y)}."

    # Check that consecutive waypoints follow four-directional movement.
    for index in range(1, len(path)):

        # Get the previous and current waypoint.
        previous = path[index - 1]
        current = path[index]

        # Calculate the change in each coordinate.
        dx = abs(current[0] - previous[0])
        dy = abs(current[1] - previous[1])

        # A legal move changes exactly one coordinate by one cell.
        if dx + dy != 1:
            return False, (
                f"Invalid movement between {previous} and {current}."
            )

    # Return success only after every check passes.
    return True, "Path passed all safety checks."


# Test the validator independently when this file is executed.
if __name__ == "__main__":

    # Import the A* planner created in the previous step.
    from path_planner import find_path

    # Define the example mission.
    start = (0, 0)
    goal = (9, 7)
    obstacles = [(2, 0), (2, 1), (2, 2)]

    # Generate a route using A*.
    path = find_path(start, goal, obstacles, grid_size=10)

    # Validate the generated route independently.
    is_safe, message = validate_path(
        path=path,
        start=start,
        goal=goal,
        obstacles=obstacles,
        grid_size=10
    )

    # Display the route and validation result.
    print("Planned path:", path)
    print("Safety status:", "PASS" if is_safe else "FAIL")
    print("Validation message:", message)
