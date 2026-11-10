
"""
route_explainer.py

Find and compare valid simple UAV routes.

Optimization priorities:
1. Minimize the number of moves.
2. Among routes with the same move count, minimize direction changes.

All route metrics are calculated deterministically in Python.
The LLM should explain these calculated results rather than invent metrics.
"""

import heapq
import itertools


def compare_route_properties(route, obstacles, start, goal):
    """Calculate measurable properties for a route."""

    route = [tuple(point) for point in route]
    obstacles = {tuple(cell) for cell in obstacles}

    directions = [
        (
            route[i][0] - route[i - 1][0],
            route[i][1] - route[i - 1][1],
        )
        for i in range(1, len(route))
    ]

    # A direction change occurs when consecutive movement vectors differ.
    direction_changes = sum(
        directions[i] != directions[i - 1]
        for i in range(1, len(directions))
    )

    return {
        "start": route[0] if route else tuple(start),
        "goal": route[-1] if route else tuple(goal),
        "moves": max(0, len(route) - 1),
        "waypoints": len(route),
        "direction_changes": direction_changes,
        "horizontal_moves": sum(dx != 0 for dx, dy in directions),
        "vertical_moves": sum(dy != 0 for dx, dy in directions),
        "obstacle_collisions": sum(point in obstacles for point in route),
    }


def build_route_explanation(
    selected_route,
    analysis,
    obstacles,
    start,
    goal,
):
    """
    Build a factual explanation using calculated route metrics.

    This function does not require an LLM and avoids unsupported claims.
    """

    selected = compare_route_properties(
        selected_route, obstacles, start, goal
    )

    lower_bound = (
        abs(goal[0] - start[0]) + abs(goal[1] - start[1])
    )

    routes = analysis.get("routes", [])
    complete = analysis.get("complete", False)

    lines = [
        f"Selected route: {selected['moves']} moves and "
        f"{selected['waypoints']} waypoints.",
        f"Direction changes: {selected['direction_changes']}.",
        f"Horizontal moves: {selected['horizontal_moves']}; "
        f"vertical moves: {selected['vertical_moves']}.",
        f"Obstacle collisions: {selected['obstacle_collisions']}.",
        f"Manhattan lower bound: {lower_bound} moves.",
    ]

    if selected["moves"] == lower_bound:
        lines.append(
            "The selected route reaches the Manhattan lower bound, "
            "so it is a shortest possible route for four-direction "
            "unit-cost movement."
        )
    else:
        lines.append(
            "The selected route exceeds the Manhattan lower bound. "
            "Obstacles may require a detour; compare it with other "
            "valid routes before claiming global optimality."
        )

    if routes:
        best_moves = min(route["moves"] for route in routes)

        best_turns = min(
            route["direction_changes"]
            for route in routes
            if route["moves"] == best_moves
        )

        lines.append(
            f"Alternative analysis returned {len(routes)} route(s). "
            f"The best move count among those routes is {best_moves}; "
            f"the fewest direction changes among routes with that "
            f"move count is {best_turns}."
        )

        if (
            selected["moves"] == best_moves
            and selected["direction_changes"] == best_turns
        ):
            lines.append(
                "The selected route matches the best move-count and "
                "direction-change metrics among the returned routes."
            )

    if complete:
        lines.append(
            "The alternative-route search exhausted its search queue."
        )
    else:
        lines.append(
            "The alternative-route search stopped before exhausting "
            "its search queue. Additional routes may exist, so the "
            "returned candidates are not an exhaustive comparison."
        )

    return " ".join(lines)


def analyze_alternative_routes(
    start,
    goal,
    obstacles,
    grid_size=10,
    max_routes=3,
    max_expansions=100_000,
):
    """
    Return up to max_routes simple routes.

    Routes are prioritized by:
      1. Estimated total moves (A* priority).
      2. Moves so far.
      3. Direction changes so far.

    The search may stop at max_routes or max_expansions. In that case,
    complete=False means the search queue was not exhausted.
    """

    start = tuple(start)
    goal = tuple(goal)
    obstacles = {tuple(cell) for cell in obstacles}

    def result(routes=None, complete=True, message="", expanded=0):
        return {
            "routes": routes or [],
            "complete": complete,
            "message": message,
            "expanded_paths": expanded,
        }

    if grid_size <= 0:
        return result(message="Grid size must be positive.")

    if max_routes <= 0:
        return result(message="max_routes must be greater than zero.")

    if max_expansions < 0:
        return result(message="max_expansions cannot be negative.")

    def in_bounds(point):
        x, y = point
        return 0 <= x < grid_size and 0 <= y < grid_size

    def heuristic(point):
        return abs(point[0] - goal[0]) + abs(point[1] - goal[1])

    if not in_bounds(start) or not in_bounds(goal):
        return result(message="Start or goal is outside the grid.")

    if start in obstacles or goal in obstacles:
        return result(message="Start or goal is blocked by an obstacle.")

    if start == goal:
        properties = compare_route_properties(
            [start], obstacles, start, goal
        )
        properties.update({
            "route": [start],
            "route_number": 1,
            "is_shortest_found": True,
            "extra_moves_vs_best_found": 0,
        })
        return result(
            routes=[properties],
            complete=True,
            message="Start and goal are the same cell.",
        )

    # Queue entries:
    # (estimated total moves, moves so far, turns so far, counter, path)
    queue = []
    counter = itertools.count()
    first_path = (start,)

    heapq.heappush(
        queue,
        (heuristic(start), 0, 0, next(counter), first_path),
    )

    found_routes = []
    expanded_paths = 0

    while queue and len(found_routes) < max_routes:
        if expanded_paths >= max_expansions:
            break

        _, moves, turns, _, path = heapq.heappop(queue)
        current = path[-1]

        if current == goal:
            properties = compare_route_properties(
                list(path), obstacles, start, goal
            )
            properties["route"] = list(path)
            found_routes.append(properties)
            continue

        expanded_paths += 1
        visited = set(path)
        x, y = current

        # Fixed neighbor order keeps repeated runs deterministic.
        neighbors = [
            (x + 1, y),
            (x, y + 1),
            (x - 1, y),
            (x, y - 1),
        ]

        previous_direction = None
        if len(path) >= 2:
            previous_direction = (
                current[0] - path[-2][0],
                current[1] - path[-2][1],
            )

        for neighbor in neighbors:
            if not in_bounds(neighbor):
                continue
            if neighbor in obstacles or neighbor in visited:
                continue

            direction = (
                neighbor[0] - current[0],
                neighbor[1] - current[1],
            )

            next_moves = moves + 1
            next_turns = turns + int(
                previous_direction is not None
                and direction != previous_direction
            )

            new_path = path + (neighbor,)

            heapq.heappush(
                queue,
                (
                    next_moves + heuristic(neighbor),
                    next_moves,
                    next_turns,
                    next(counter),
                    new_path,
                ),
            )

    # A nonempty queue means candidates may remain unexplored.
    complete = not queue

    # Sort explicitly so returned results obey the same ranking rule.
    found_routes.sort(
        key=lambda item: (
            item["moves"],
            item["direction_changes"],
            tuple(item["route"]),
        )
    )

    if found_routes:
        best_key = (
            found_routes[0]["moves"],
            found_routes[0]["direction_changes"],
        )

        for index, item in enumerate(found_routes, start=1):
            item["route_number"] = index
            item["is_shortest_found"] = (
                item["moves"] == found_routes[0]["moves"]
            )
            item["is_best_found"] = (
                item["moves"], item["direction_changes"]
            ) == best_key
            item["extra_moves_vs_best_found"] = (
                item["moves"] - found_routes[0]["moves"]
            )

        message = (
            f"Found {len(found_routes)} valid route(s). "
            f"Best returned route: {best_key[0]} moves and "
            f"{best_key[1]} direction changes."
        )
    else:
        message = "No valid route was found within the search budget."

    if not complete:
        message += (
            " Search stopped before exhausting all candidates; "
            "additional routes may exist."
        )
    else:
        message += " The search queue was exhausted."

    return result(
        routes=found_routes,
        complete=complete,
        message=message,
        expanded=expanded_paths,
    )
