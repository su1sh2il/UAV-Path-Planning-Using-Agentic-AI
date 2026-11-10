
from path_planner import a_star

start = (0, 0)
goal = (9, 7)
obstacles = [(2, 0), (2, 1), (2, 2)]

route = a_star(start, goal, obstacles, grid_size=10)

def count_direction_changes(route):
    changes = 0
    previous_direction = None

    for i in range(1, len(route)):
        dx = route[i][0] - route[i - 1][0]
        dy = route[i][1] - route[i - 1][1]
        direction = (dx, dy)

        if previous_direction is not None and direction != previous_direction:
            changes += 1

        previous_direction = direction

    return changes

print("Route:", route)
print("Moves:", len(route) - 1)
print("Direction changes:", count_direction_changes(route))
print("Obstacle collisions:", any(point in obstacles for point in route))

assert route is not None
assert len(route) - 1 == 16
assert count_direction_changes(route) == 1
assert not any(point in obstacles for point in route)

print("All tests passed!")
