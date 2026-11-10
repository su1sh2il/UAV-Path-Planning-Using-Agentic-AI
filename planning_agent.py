
"""
planning_agent.py

Existing Microsoft Agent Framework UAV agent, extended with
a route explanation tool.

Tools:
1. plan_uav_route      - calculate a route using A*
2. validate_uav_route  - independently validate the route
3. explain_uav_route   - provide verified facts for route explanation
"""

import json

from agent_framework import Agent
from agent_framework.ollama import OllamaChatClient

from path_planner import find_path
from safety_validator import validate_path
from route_explainer import analyze_alternative_routes


async def plan_mission_with_agent(
    start,
    goal,
    obstacles,
    grid_size=10,
):
    """Plan, validate, and explain a UAV mission."""

    # Store actual Python results. Never trust the LLM alone
    # to decide whether a route exists or is safe.
    state = {
        "route": None,
        "planning_called": False,
        "validation_called": False,
        "explanation_called": False,
        "safe": False,
        "message": "Mission not completed.",
        "explanation_facts": None,
    }

    start = tuple(start)
    goal = tuple(goal)
    obstacles = {tuple(cell) for cell in obstacles}

    # -----------------------------------------------------
    # TOOL 1: Plan the route using the existing A* function
    # -----------------------------------------------------
    def plan_uav_route() -> str:
        """Calculate a route with A*. Call this first."""

        state["planning_called"] = True

        route = find_path(
            start=start,
            goal=goal,
            obstacles=obstacles,
            grid_size=grid_size,
        )

        if not route:
            state["route"] = None
            state["safe"] = False
            state["message"] = (
                "No route exists for the supplied mission."
            )

            return json.dumps({
                "success": False,
                "message": state["message"],
                "route": None,
            })

        # Save the actual route returned by A*.
        state["route"] = [tuple(point) for point in route]
        state["message"] = "A* found a candidate route."

        return json.dumps({
            "success": True,
            "route": state["route"],
            "moves": len(route) - 1,
            "message": (
                "Route found. Call validate_uav_route next."
            ),
        })

    # -----------------------------------------------------
    # TOOL 2: Independently validate the calculated route
    # -----------------------------------------------------
    def validate_uav_route() -> str:
        """Validate the planned route before approval."""

        state["validation_called"] = True
        route = state["route"]

        if not route:
            state["safe"] = False
            state["message"] = (
                "Cannot validate because no route exists."
            )

            return json.dumps({
                "safe": False,
                "message": state["message"],
            })

        safe, message = validate_path(
            path=route,
            start=start,
            goal=goal,
            obstacles=obstacles,
            grid_size=grid_size,
        )

        state["safe"] = bool(safe)
        state["message"] = message

        return json.dumps({
            "safe": bool(safe),
            "message": message,
            "instruction": (
                "Call explain_uav_route next."
            ),
        })

    # -----------------------------------------------------
    # TOOL 3: Calculate facts that support an explanation
    # -----------------------------------------------------
    def explain_uav_route() -> str:
        """
        Calculate evidence for explaining why the route works.
        Call only after planning and safety validation.
        """

        if not state["route"]:
            return json.dumps({
                "available": False,
                "reason": "No route exists to explain.",
            })

        if not state["validation_called"] or not state["safe"]:
            return json.dumps({
                "available": False,
                "reason": (
                    "The route has not passed safety validation."
                ),
            })

        state["explanation_called"] = True
        route = state["route"]
        moves = len(route) - 1

        # Manhattan distance is a lower bound for four-direction
        # movement when each move costs one.
        lower_bound = (
            abs(goal[0] - start[0])
            + abs(goal[1] - start[1])
        )

        # Verify that the route uses only four-direction moves.
        invalid_steps = []
        for previous, current in zip(route, route[1:]):
            distance = (
                abs(current[0] - previous[0])
                + abs(current[1] - previous[1])
            )
            if distance != 1:
                invalid_steps.append([previous, current])

        # Identify obstacles lying directly on the route's
        # horizontal or vertical line between start and goal.
        direct_blockers = []

        if start[1] == goal[1]:
            for x in range(
                min(start[0], goal[0]),
                max(start[0], goal[0]) + 1,
            ):
                if (x, start[1]) in obstacles:
                    direct_blockers.append((x, start[1]))

        elif start[0] == goal[0]:
            for y in range(
                min(start[1], goal[1]),
                max(start[1], goal[1]) + 1,
            ):
                if (start[0], y) in obstacles:
                    direct_blockers.append((start[0], y))


        # Calculate alternative valid routes using Python.
        alternative_analysis = analyze_alternative_routes(
            start=start,
            goal=goal,
            obstacles=obstacles,
            grid_size=grid_size,
            max_routes=3,
        )

        facts = {
            "available": True,
            "start": start,
            "goal": goal,
            "grid_size": grid_size,
            "obstacle_count": len(obstacles),
            "obstacles": sorted(obstacles),
            "route": route,
            "waypoint_count": len(route),
            "move_count": moves,
            "manhattan_lower_bound": lower_bound,
            "matches_lower_bound": moves == lower_bound,
            "extra_moves_over_lower_bound": moves - lower_bound,
            "safety_validated": state["safe"],
            "safety_message": state["message"],
            "route_contains_invalid_steps": bool(invalid_steps),
            "invalid_steps": invalid_steps,
            "direct_line_blockers": direct_blockers,
            "movement_rule": (
                "Four directions only; one cell per move."
            ),

            "alternative_routes": alternative_analysis["routes"],
            "alternative_search_complete": alternative_analysis["complete"],
            "alternative_search_message": alternative_analysis["message"],
            "alternative_paths_expanded": alternative_analysis["expanded_paths"],
        }

        state["explanation_facts"] = facts

        return json.dumps(facts)

    # -----------------------------------------------------
    # Use the same Microsoft Agent Framework and Ollama model
    # -----------------------------------------------------
    client = OllamaChatClient(model="qwen3:4b")

    uav_agent = Agent(
        client=client,
        name="UAVMissionPlanner",
        instructions=(
            "You coordinate a UAV grid-planning mission. "
            "For each mission, call plan_uav_route first. "
            "If a route exists, call validate_uav_route. "
            "Only after validation succeeds, call explain_uav_route. "
            "Explain why the route works using the tool results. "
            "Discuss obstacle avoidance, move count, and the "
            "Manhattan lower bound when the facts support it. "
            "Explain that a route matching the lower bound is "
            "shortest under four-direction unit-cost assumptions. "
            "Do not claim to have compared or searched alternative "
            "routes unless the tools provide that evidence. "
            "Never invent coordinates or safety results. "
            "If no route exists, explain the planning failure "
            "without claiming the mission is safe. "
            "Keep the final explanation clear and concise."

            "Use the alternative-route analysis when explaining route choices. "
            "Compare actual returned routes and their move counts. "
            "Do not claim a route is uniquely optimal when other routes tie. "
            "Do not claim every possible route was examined unless the "
            "alternative search explicitly establishes that conclusion. "

            "Compare the selected route with the alternative routes using "
            "the actual route-analysis results. "
            "Explain the differences in move count, direction changes, "
            "horizontal moves, vertical moves, and obstacle collisions. "
            "If multiple routes have the same minimum move count, state "
            "that they are equally optimal under four-direction "
            "unit-cost movement. "
            "Explain the actual reason the planner returned its selected "
            "route, including its deterministic tie-breaking rule if known. "
            "Do not claim the selected route is safer, shorter, or more "
            "energy-efficient unless the calculated evidence supports it. "
            "If the evidence does not establish a unique best route, "
            "say that the alternatives are equally good under the "
            "criteria currently evaluated. "


        ),
        tools=[
            plan_uav_route,
            validate_uav_route,
            explain_uav_route,
        ],
    )

    mission = f"""
    Explain and plan this UAV mission.

    Grid: {grid_size} x {grid_size}
    Start: {start}
    Goal: {goal}
    Obstacles: {sorted(obstacles)}
    Movement: four directions, one cell per move.

    Plan, validate, and explain the route using the tools.
    """

    response = await uav_agent.run(mission)
    agent_summary = response.text or ""

    # Trust actual tool results rather than claims in model text.
    if not state["planning_called"]:
        return {
            "route": None,
            "safe": False,
            "message": "Agent did not call the planning tool.",
            "agent_summary": agent_summary,
        }

    if not state["route"]:
        return {
            "route": None,
            "safe": False,
            "message": state["message"],
            "agent_summary": agent_summary,
        }

    if not state["validation_called"] or not state["safe"]:
        return {
            "route": None,
            "safe": False,
            "message": (
                "Route was not approved by the safety validator."
            ),
            "agent_summary": agent_summary,
        }

    # If the LLM skips the explanation tool, return verified
    # facts so the dashboard can still present an explanation.
    if not state["explanation_called"]:
        facts = explain_uav_route()
        state["explanation_called"] = True

    return {
        "route": state["route"],
        "safe": True,
        "message": state["message"],
        "agent_summary": agent_summary,
        "explanation_facts": state["explanation_facts"],
    }
