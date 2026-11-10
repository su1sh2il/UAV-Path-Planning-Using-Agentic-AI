
# Import the A* path-planning function.
from path_planner import find_path

# Import the independent safety validator.
from safety_validator import validate_path


# Define a tool that computes a route for a UAV mission.
def plan_uav_path(
    start_x: int,
    start_y: int,
    goal_x: int,
    goal_y: int,
    obstacles: list,
    grid_size: int = 10
) -> dict:
    """Calculate a four-directional A* route around obstacle cells."""

    # Convert the supplied coordinates into tuples.
    start = (start_x, start_y)
    goal = (goal_x, goal_y)

    # Ask the deterministic A* planner to calculate the route.
    path = find_path(start, goal, obstacles, grid_size)

    # Return a structured result if no route can be found.
    if path is None:
        return {
            "success": False,
            "path": [],
            "message": "No valid route was found."
        }

    # Return the computed route and its move count.
    return {
        "success": True,
        "path": [list(point) for point in path],
        "number_of_moves": len(path) - 1,
        "message": "Route calculated; safety validation is still required."
    }


# Define a tool that checks a proposed route independently.
def validate_uav_path(
    path: list,
    start_x: int,
    start_y: int,
    goal_x: int,
    goal_y: int,
    obstacles: list,
    grid_size: int = 10
) -> dict:
    """Validate grid bounds, obstacles, endpoints, and movement rules."""

    # Convert coordinates to tuples for the validator.
    start = (start_x, start_y)
    goal = (goal_x, goal_y)

    # Independently validate the proposed route.
    is_safe, message = validate_path(
        path=path,
        start=start,
        goal=goal,
        obstacles=obstacles,
        grid_size=grid_size
    )

    # Return a structured validation result.
    return {
        "safe": is_safe,
        "message": message
    }
