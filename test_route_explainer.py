
from route_explainer import (
    analyze_alternative_routes,
    build_route_explanation,
    compare_route_properties,
)

start = (0, 0)
goal = (9, 7)
obstacles = [(2, 0), (2, 1), (2, 2)]

selected_route = [
    (0, 0), (0, 1), (0, 2), (0, 3),
    (0, 4), (0, 5), (0, 6), (0, 7),
    (1, 7), (2, 7), (3, 7), (4, 7),
    (5, 7), (6, 7), (7, 7), (8, 7), (9, 7),
]

metrics = compare_route_properties(
    selected_route, obstacles, start, goal
)

assert metrics["moves"] == 16
assert metrics["waypoints"] == 17
assert metrics["direction_changes"] == 1
assert metrics["obstacle_collisions"] == 0

analysis = analyze_alternative_routes(
    start=start,
    goal=goal,
    obstacles=obstacles,
    grid_size=10,
    max_routes=3,
)

for route in analysis["routes"]:
    assert route["obstacle_collisions"] == 0

    # Each alternative's metrics must match its coordinates.
    recalculated = compare_route_properties(
        route["route"], obstacles, start, goal
    )
    assert route["moves"] == recalculated["moves"]
    assert route["direction_changes"] == recalculated["direction_changes"]

print("Selected route metrics:", metrics)
print("Alternative search:", analysis["message"])

print(
    build_route_explanation(
        selected_route, analysis, obstacles, start, goal
    )
)

print("All route-explainer tests passed!")
