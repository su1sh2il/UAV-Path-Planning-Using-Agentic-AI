
"""
UAV Path Planning using A* Search.

Optimization priorities:
1. Minimize the number of moves.
2. Among equally short routes, minimize direction changes.

Coordinates use the format (x, y).
Movement is allowed in four directions only.
"""

import heapq
from itertools import count


def a_star(start, goal, obstacles, grid_size=10):
    """
    Find a shortest collision-free route with the fewest turns.

    Returns:
        List of coordinates from start to goal, or None if no route exists.
    """

    start = tuple(start)
    goal = tuple(goal)
    blocked = {tuple(point) for point in obstacles}

    if grid_size <= 0:
        raise ValueError("grid_size must be positive.")

    def in_bounds(point):
        x, y = point
        return 0 <= x < grid_size and 0 <= y < grid_size

    if not in_bounds(start) or not in_bounds(goal):
        raise ValueError("Start and goal must be inside the grid.")

    if start in blocked or goal in blocked:
        return None

    if start == goal:
        return [start]

    # Each direction has a unique index.
    directions = [
        (0, 1),    # Up
        (1, 0),    # Right
        (0, -1),   # Down
        (-1, 0),   # Left
    ]

    def heuristic(point):
        """Manhattan distance: minimum remaining moves."""
        return abs(point[0] - goal[0]) + abs(point[1] - goal[1])

    # State includes the previous direction because turn cost depends
    # on how the UAV entered the current cell.
    initial_state = (start, None)

    # Heap order:
    # (estimated total moves, turns so far, moves so far, counter, state)
    sequence = count()
    open_heap = [
        (heuristic(start), 0, 0, next(sequence), initial_state)
    ]

    # Store the best (moves, turns) for each coordinate/direction state.
    best_cost = {initial_state: (0, 0)}

    # Parent links are used to reconstruct the route.
    parent = {}

    while open_heap:
        _, turns, moves, _, state = heapq.heappop(open_heap)

        point, previous_direction = state
        current_cost = (moves, turns)

        # Skip outdated heap entries.
        if current_cost != best_cost.get(state):
            continue

        # The priority ordering ensures the first goal reached when
        # popped is optimal by moves first, then direction changes.
        if point == goal:
            route = []
            current_state = state

            while current_state is not None:
                route.append(current_state[0])
                current_state = parent.get(current_state)

            route.reverse()
            return route

        # Expand all four neighboring cells.
        for direction_index, (dx, dy) in enumerate(directions):
            neighbor = (point[0] + dx, point[1] + dy)

            # Reject cells outside the grid or occupied by obstacles.
            if not in_bounds(neighbor) or neighbor in blocked:
                continue

            next_moves = moves + 1

            # The first move is not a direction change.
            direction_changed = (
                previous_direction is not None
                and direction_index != previous_direction
            )
            next_turns = turns + int(direction_changed)

            next_state = (neighbor, direction_index)
            next_cost = (next_moves, next_turns)

            # Lexicographic comparison prioritizes fewer moves,
            # then fewer turns when move counts are equal.
            old_cost = best_cost.get(
                next_state,
                (float("inf"), float("inf"))
            )

            if next_cost < old_cost:
                best_cost[next_state] = next_cost
                parent[next_state] = state

                heapq.heappush(
                    open_heap,
                    (
                        next_moves + heuristic(neighbor),
                        next_turns,
                        next_moves,
                        next(sequence),
                        next_state,
                    ),
                )

    # The destination cannot be reached.
    return None


# Compatibility wrappers for existing imports.
def a_star_search(start, goal, obstacles, grid_size=10):
    return a_star(start, goal, obstacles, grid_size)


def find_path(start, goal, obstacles, grid_size=10):
    return a_star(start, goal, obstacles, grid_size)


def plan_path(start, goal, obstacles, grid_size=10):
    return a_star(start, goal, obstacles, grid_size)
