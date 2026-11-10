
"""
UAV Path Planning Dashboard — Hybrid Agentic AI Mode

Fast Mode:
    1. Plan using deterministic A*.
    2. Independently validate the route.
    3. Display only a validated route.

AI-Assisted Mode:
    1. Interpret the mission using Microsoft Agent Framework
       and the local Ollama Qwen3:4b model.
    2. Obtain a route from the planning agent.
    3. Independently validate the returned route.
    4. Display only a validated route.

Dashboard features:
    - Interactive 2D grid.
    - Immediate map updates when inputs change.
    - Route planning only after clicking the button.
    - Route metrics and deterministic explanations.
    - Alternative-route analysis and ranking.
    - Explicit handling of incomplete alternative searches.
"""

import asyncio

import matplotlib.pyplot as plt
import streamlit as st

from path_planner import find_path
from safety_validator import validate_path
from planning_agent import plan_mission_with_agent

from route_explainer import (
    analyze_alternative_routes,
    compare_route_properties,
    build_route_explanation,
)


# ==================================================
# 1. APPLICATION CONFIGURATION
# ==================================================

GRID_SIZE = 10
MAX_ALTERNATIVE_ROUTES = 3

FAST_MODE = "Fast Mode (A* + Safety)"
AI_MODE = "AI-Assisted Mode (Microsoft Agent Framework)"

st.set_page_config(
    page_title="UAV Agentic Path Planning",
    page_icon="🚁",
    layout="wide",
)

st.title("🚁 UAV Path Planning Using Agentic AI")

st.caption(
    "Hybrid architecture: deterministic A* path planning, "
    "independent safety validation, and optional AI assistance."
)


# ==================================================
# 2. INITIALIZE SESSION STATE
# ==================================================

# Keep the route between Streamlit reruns.
if "planned_route" not in st.session_state:
    st.session_state.planned_route = None

# Store the inputs used for the last planning attempt.
if "planned_inputs" not in st.session_state:
    st.session_state.planned_inputs = None

# Store the last planning or validation message.
if "planning_message" not in st.session_state:
    st.session_state.planning_message = ""

# Store the AI agent's interpretation, if available.
if "agent_summary" not in st.session_state:
    st.session_state.agent_summary = ""

# Store which planning mode produced the result.
if "result_mode" not in st.session_state:
    st.session_state.result_mode = ""


# ==================================================
# 3. HELPER: PARSE OBSTACLE COORDINATES
# ==================================================

def parse_obstacles(text):
    """
    Convert obstacle text into a list of (x, y) tuples.

    Example input:
        2,0; 2,1; 2,2

    Empty input is allowed and means there are no obstacles.
    Duplicate coordinates are removed.
    Coordinates outside the grid are rejected.
    """

    obstacles = []

    if not text.strip():
        return obstacles

    for item in text.split(";"):
        item = item.strip()

        # Ignore empty entries, including a trailing semicolon.
        if not item:
            continue

        parts = item.split(",")

        if len(parts) != 2:
            raise ValueError(
                f"Invalid obstacle '{item}'. "
                "Use x,y format, for example 2,3."
            )

        try:
            x = int(parts[0].strip())
            y = int(parts[1].strip())

        except ValueError as error:
            raise ValueError(
                f"Invalid obstacle '{item}'. "
                "Coordinates must be integers."
            ) from error

        # Ensure the obstacle is inside the grid.
        if not (
            0 <= x < GRID_SIZE
            and 0 <= y < GRID_SIZE
        ):
            raise ValueError(
                f"Obstacle ({x}, {y}) is outside the "
                f"{GRID_SIZE}x{GRID_SIZE} grid."
            )

        # Keep each obstacle only once.
        if (x, y) not in obstacles:
            obstacles.append((x, y))

    return obstacles


# ==================================================
# 4. HELPER: CLEAR PREVIOUS RESULTS
# ==================================================

def clear_planning_results():
    """
    Clear the previous route and associated explanations.

    This prevents an old route from being displayed after
    the mission configuration changes.
    """

    st.session_state.planned_route = None
    st.session_state.planned_inputs = None
    st.session_state.planning_message = ""
    st.session_state.agent_summary = ""
    st.session_state.result_mode = ""


# ==================================================
# 5. HELPER: INDEPENDENT ROUTE VALIDATION
# ==================================================

def independently_validate_route(route, start, goal, obstacles):
    """
    Validate a route before displaying it.

    The safety validator checks the route according to
    the project's existing validation rules.

    This function also checks basic route structure and
    confirms that the route begins and ends correctly.
    """

    if not route:
        return False, "No route was returned by the planner."

    try:
        # Convert coordinates into tuples for consistent comparison.
        normalized_route = [
            tuple(point) for point in route
        ]

    except (TypeError, ValueError):
        return False, "The planner returned an invalid route format."

    # Every waypoint must contain exactly two integer coordinates.
    for point in normalized_route:

        if len(point) != 2:
            return False, f"Invalid waypoint format: {point}"

        x, y = point

        # Reject Boolean values even though bool is a subclass of int.
        if (
            isinstance(x, bool)
            or isinstance(y, bool)
            or not isinstance(x, int)
            or not isinstance(y, int)
        ):
            return False, f"Waypoint coordinates must be integers: {point}"

        if not (
            0 <= x < GRID_SIZE
            and 0 <= y < GRID_SIZE
        ):
            return False, f"Waypoint {point} is outside the grid."

    # The first and final waypoints must match the mission.
    if normalized_route[0] != start:
        return False, "The route does not start at the requested start."

    if normalized_route[-1] != goal:
        return False, "The route does not end at the requested goal."

    # A route must not visit an obstacle.
    obstacle_set = set(obstacles)

    for point in normalized_route:
        if point in obstacle_set:
            return False, f"The route passes through obstacle {point}."

    # Every consecutive pair must be a legal four-direction move.
    for current, next_point in zip(
        normalized_route,
        normalized_route[1:],
    ):
        distance = (
            abs(next_point[0] - current[0])
            + abs(next_point[1] - current[1])
        )

        if distance != 1:
            return False, (
                f"Illegal move from {current} to {next_point}. "
                "Each move must go one cell horizontally or vertically."
            )

    # Call the project's independent safety validator.
    safe, validation_message = validate_path(
        path=normalized_route,
        start=start,
        goal=goal,
        obstacles=obstacles,
        grid_size=GRID_SIZE,
    )

    if not safe:
        return False, str(validation_message)

    return True, str(validation_message)


# ==================================================
# 6. SIDEBAR: MISSION CONFIGURATION
# ==================================================

st.sidebar.header("Mission Configuration")

st.sidebar.subheader("Start Position")

start_x = st.sidebar.slider(
    "Start X",
    min_value=0,
    max_value=GRID_SIZE - 1,
    value=0,
)

start_y = st.sidebar.slider(
    "Start Y",
    min_value=0,
    max_value=GRID_SIZE - 1,
    value=0,
)

st.sidebar.subheader("Goal Position")

goal_x = st.sidebar.slider(
    "Goal X",
    min_value=0,
    max_value=GRID_SIZE - 1,
    value=9,
)

goal_y = st.sidebar.slider(
    "Goal Y",
    min_value=0,
    max_value=GRID_SIZE - 1,
    value=7,
)

obstacle_text = st.sidebar.text_input(
    "Obstacle coordinates",
    value="2,0; 2,1; 2,2",
    help=(
        "Enter x,y coordinates separated by semicolons. "
        "Example: 2,0; 3,1; 4,2"
    ),
)

st.sidebar.subheader("Planning Mode")

planning_mode = st.sidebar.radio(
    "Choose how to plan",
    options=[
        FAST_MODE,
        AI_MODE,
    ],
    index=0,
    help=(
        "Fast Mode uses A* directly. AI-Assisted Mode calls "
        "the local Ollama model through Microsoft Agent Framework."
    ),
)

plan_clicked = st.sidebar.button(
    "Plan and Validate Route",
    type="primary",
    use_container_width=True,
)


# ==================================================
# 7. READ AND VALIDATE CURRENT INPUTS
# ==================================================

start = (start_x, start_y)
goal = (goal_x, goal_y)

input_error = ""

try:
    obstacles = parse_obstacles(obstacle_text)

except ValueError as error:
    # Keep the grid visible even if obstacle text is invalid.
    obstacles = []
    input_error = str(error)

# Check that start and goal are different.
if not input_error and start == goal:
    input_error = (
        "Start and goal must be different positions."
    )

# The UAV cannot start inside an obstacle.
if not input_error and start in obstacles:
    input_error = (
        "The start position cannot be an obstacle."
    )

# The destination cannot be an obstacle.
if not input_error and goal in obstacles:
    input_error = (
        "The goal position cannot be an obstacle."
    )

# Include all mission settings in the input fingerprint.
# Changing any of these settings invalidates the old result.
current_inputs = (
    start,
    goal,
    tuple(obstacles),
    input_error,
    planning_mode,
)


# ==================================================
# 8. INVALIDATE STALE RESULTS
# ==================================================

if current_inputs != st.session_state.planned_inputs:
    clear_planning_results()


# ==================================================
# 9. PLAN ONLY AFTER THE BUTTON IS CLICKED
# ==================================================

if plan_clicked:

    if input_error:
        clear_planning_results()

        st.session_state.planned_inputs = current_inputs
        st.session_state.planning_message = input_error

    else:
        try:
            # ------------------------------------------
            # 9A. FAST MODE
            # ------------------------------------------

            if planning_mode == FAST_MODE:

                with st.spinner(
                    "A* is calculating the route and "
                    "the safety validator is checking it..."
                ):

                    # Calculate the route using deterministic A*.
                    candidate_route = find_path(
                        start=start,
                        goal=goal,
                        obstacles=obstacles,
                        grid_size=GRID_SIZE,
                    )

                    if not candidate_route:

                        result = {
                            "route": None,
                            "message": (
                                "No valid route was found. "
                                "Try changing the obstacles or destination."
                            ),
                            "agent_summary": "",
                        }

                    else:
                        # Independently validate the returned route.
                        safe, validation_message = (
                            independently_validate_route(
                                route=candidate_route,
                                start=start,
                                goal=goal,
                                obstacles=obstacles,
                            )
                        )

                        result = {
                            "route": (
                                [tuple(point) for point in candidate_route]
                                if safe
                                else None
                            ),
                            "message": (
                                f"Route found and validated successfully. "
                                f"{validation_message}"
                                if safe
                                else (
                                    "Route rejected by safety validation: "
                                    f"{validation_message}"
                                )
                            ),
                            "agent_summary": "",
                        }

            # ------------------------------------------
            # 9B. AI-ASSISTED MODE
            # ------------------------------------------

            else:

                with st.spinner(
                    "Microsoft Agent Framework is interpreting "
                    "the mission and requesting a route from "
                    "the local AI agent..."
                ):

                    # Run the existing asynchronous agent function.
                    agent_result = asyncio.run(
                        plan_mission_with_agent(
                            start=start,
                            goal=goal,
                            obstacles=obstacles,
                            grid_size=GRID_SIZE,
                        )
                    )

                    # Ensure the agent returned the expected structure.
                    if not isinstance(agent_result, dict):
                        raise TypeError(
                            "The planning agent must return a dictionary."
                        )

                    candidate_route = agent_result.get("route")

                    if not candidate_route:

                        result = {
                            "route": None,
                            "message": agent_result.get(
                                "message",
                                "The AI-assisted planner did not return a route.",
                            ),
                            "agent_summary": agent_result.get(
                                "agent_summary",
                                "",
                            ),
                        }

                    else:
                        # Do not trust the agent's safety claim alone.
                        # Recheck its returned route independently.
                        safe, validation_message = (
                            independently_validate_route(
                                route=candidate_route,
                                start=start,
                                goal=goal,
                                obstacles=obstacles,
                            )
                        )

                        result = {
                            "route": (
                                [tuple(point) for point in candidate_route]
                                if safe
                                else None
                            ),
                            "message": (
                                "AI-assisted route independently validated. "
                                f"{validation_message}"
                                if safe
                                else (
                                    "AI-assisted route rejected by "
                                    "independent safety validation: "
                                    f"{validation_message}"
                                )
                            ),
                            "agent_summary": agent_result.get(
                                "agent_summary",
                                "",
                            ),
                        }

            # ------------------------------------------
            # 9C. SAVE THE RESULT
            # ------------------------------------------

            st.session_state.planned_route = result["route"]
            st.session_state.planned_inputs = current_inputs
            st.session_state.planning_message = result["message"]
            st.session_state.agent_summary = result.get(
                "agent_summary",
                "",
            )
            st.session_state.result_mode = planning_mode

        except Exception as error:
            # Report the error without crashing the dashboard.
            clear_planning_results()

            st.session_state.planned_inputs = current_inputs
            st.session_state.planning_message = (
                f"Planning failed: {type(error).__name__}: {error}"
            )
            st.session_state.result_mode = planning_mode


# ==================================================
# 10. DISPLAY PLANNING STATUS
# ==================================================

if input_error:

    st.warning(input_error)

elif st.session_state.planning_message:

    if st.session_state.planned_route is not None:
        st.success(st.session_state.planning_message)

    else:
        st.warning(st.session_state.planning_message)


# ==================================================
# 11. DRAW THE LIVE 2D GRID
# ==================================================

# The grid is drawn on every rerun.
# Route planning itself still requires a button click.

fig, ax = plt.subplots(figsize=(8, 8))

ax.set_xlim(-0.5, GRID_SIZE - 0.5)
ax.set_ylim(-0.5, GRID_SIZE - 0.5)

ax.set_xticks(range(GRID_SIZE))
ax.set_yticks(range(GRID_SIZE))

ax.set_xticks(
    [i - 0.5 for i in range(GRID_SIZE + 1)],
    minor=True,
)

ax.set_yticks(
    [i - 0.5 for i in range(GRID_SIZE + 1)],
    minor=True,
)

ax.grid(
    which="minor",
    linestyle="-",
    linewidth=0.8,
)

ax.grid(
    which="major",
    visible=False,
)

ax.set_aspect("equal")
ax.set_xlabel("X Coordinate")
ax.set_ylabel("Y Coordinate")
ax.set_title("UAV 2D Grid Map")

# Draw obstacles.
if obstacles:

    obstacle_x = [point[0] for point in obstacles]
    obstacle_y = [point[1] for point in obstacles]

    ax.scatter(
        obstacle_x,
        obstacle_y,
        color="black",
        marker="s",
        s=180,
        label="Obstacles",
        zorder=3,
    )

# Draw start.
ax.scatter(
    start[0],
    start[1],
    color="green",
    marker="o",
    s=220,
    label="Start",
    zorder=5,
)

# Draw destination.
ax.scatter(
    goal[0],
    goal[1],
    color="red",
    marker="*",
    s=300,
    label="Goal",
    zorder=5,
)

# Draw only a validated route belonging to the current inputs.
route = st.session_state.planned_route

if (
    route is not None
    and st.session_state.planned_inputs == current_inputs
    and not input_error
):

    route_x = [point[0] for point in route]
    route_y = [point[1] for point in route]

    ax.plot(
        route_x,
        route_y,
        color="blue",
        linewidth=2.5,
        marker="o",
        markersize=4,
        label="Validated Route",
        zorder=4,
    )

ax.legend(
    loc="upper left",
    bbox_to_anchor=(1.02, 1.0),
)

fig.tight_layout()

st.pyplot(fig)
plt.close(fig)


# ==================================================
# 12. ROUTE RESULTS AND METRICS
# ==================================================

route = st.session_state.planned_route

if (
    route is not None
    and st.session_state.planned_inputs == current_inputs
    and not input_error
):

    st.subheader("Route Results")

    # Calculate all route metrics using Python.
    selected_metrics = compare_route_properties(
        route=route,
        obstacles=obstacles,
        start=start,
        goal=goal,
    )

    # First row: primary metrics.
    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Grid Size",
        f"{GRID_SIZE} × {GRID_SIZE}",
    )

    col2.metric(
        "Route Waypoints",
        selected_metrics["waypoints"],
    )

    col3.metric(
        "Total Moves",
        selected_metrics["moves"],
    )

    # Second row: movement characteristics.
    col4, col5, col6 = st.columns(3)

    col4.metric(
        "Direction Changes",
        selected_metrics["direction_changes"],
    )

    col5.metric(
        "Horizontal Moves",
        selected_metrics["horizontal_moves"],
    )

    col6.metric(
        "Vertical Moves",
        selected_metrics["vertical_moves"],
    )

    # Display the computed collision count.
    if selected_metrics["obstacle_collisions"] == 0:

        st.success(
            "Obstacle collisions: 0 — no obstacle collisions detected."
        )

    else:

        st.error(
            "Obstacle collisions detected: "
            f"{selected_metrics['obstacle_collisions']}"
        )

    # Display the complete waypoint sequence.
    st.write("**Selected validated route:**")

    st.code(
        " → ".join(str(point) for point in route)
    )

    # --------------------------------------------------
    # 12A. ALTERNATIVE ROUTE ANALYSIS
    # --------------------------------------------------

    with st.expander(
        "Alternative Route Analysis",
        expanded=True,
    ):

        # Candidate generation is deterministic and independent
        # of the LLM's natural-language response.
        analysis = analyze_alternative_routes(
            start=start,
            goal=goal,
            obstacles=obstacles,
            grid_size=GRID_SIZE,
            max_routes=MAX_ALTERNATIVE_ROUTES,
        )

        st.write(analysis["message"])

        alternatives = analysis["routes"]

        # Recompute each candidate's properties and validate it.
        ranked_alternatives = []

        for item in alternatives:

            candidate_route = item["route"]

            candidate_safe, candidate_message = (
                independently_validate_route(
                    route=candidate_route,
                    start=start,
                    goal=goal,
                    obstacles=obstacles,
                )
            )

            # Never display an invalid alternative as a valid route.
            if not candidate_safe:
                st.warning(
                    "An alternative candidate was excluded because "
                    f"it failed validation: {candidate_message}"
                )
                continue

            candidate_metrics = compare_route_properties(
                route=candidate_route,
                obstacles=obstacles,
                start=start,
                goal=goal,
            )

            ranked_alternatives.append({
                "route": [tuple(point) for point in candidate_route],
                "metrics": candidate_metrics,
            })

        # Rank by move count, then turns, then coordinate sequence.
        ranked_alternatives.sort(
            key=lambda item: (
                item["metrics"]["moves"],
                item["metrics"]["direction_changes"],
                tuple(item["route"]),
            )
        )

        if ranked_alternatives:

            best_moves = min(
                item["metrics"]["moves"]
                for item in ranked_alternatives
            )

            best_turns = min(
                item["metrics"]["direction_changes"]
                for item in ranked_alternatives
                if item["metrics"]["moves"] == best_moves
            )

            col1, col2 = st.columns(2)

            col1.metric(
                "Fewest Moves Among Candidates",
                best_moves,
            )

            col2.metric(
                "Fewest Turns at That Move Count",
                best_turns,
            )

            # Display each candidate and its computed properties.
            for rank, item in enumerate(
                ranked_alternatives,
                start=1,
            ):

                candidate_route = item["route"]
                metrics = item["metrics"]

                st.markdown(f"### Candidate Route {rank}")

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Moves",
                    metrics["moves"],
                )

                col2.metric(
                    "Waypoints",
                    metrics["waypoints"],
                )

                col3.metric(
                    "Direction Changes",
                    metrics["direction_changes"],
                )

                if (
                    metrics["moves"] == best_moves
                    and metrics["direction_changes"] == best_turns
                ):

                    st.write(
                        "Matches the best move-count and "
                        "direction-change metrics among the "
                        "validated candidates returned."
                    )

                elif metrics["moves"] == best_moves:

                    st.write(
                        "Ties for the fewest moves found, "
                        "but has more direction changes."
                    )

                else:

                    st.write(
                        f"Uses {metrics['moves'] - best_moves} "
                        "additional move(s) compared with the "
                        "fewest-move candidate found."
                    )

                st.code(
                    " → ".join(
                        str(point)
                        for point in candidate_route
                    )
                )

                st.divider()

        else:

            st.warning(
                "No valid alternative routes were returned."
            )

        # Do not claim that the candidate list is exhaustive
        # unless the search implementation reports completion.
        if analysis["complete"]:

            st.caption(
                "The alternative-route search queue was exhausted. "
                "The search completed under its implemented rules."
            )

        else:

            st.caption(
                "The search stopped before exhausting its queue. "
                "Additional routes may exist. Metrics describe only "
                "the validated candidates returned by the search."
            )

    # --------------------------------------------------
    # 12B. DETERMINISTIC ROUTE EXPLANATION
    # --------------------------------------------------

    st.subheader("Route Explanation")

    # Explain the selected route using computed properties and
    # the alternative-search result, not LLM-estimated metrics.
    explanation = build_route_explanation(
        selected_route=route,
        analysis=analysis,
        obstacles=obstacles,
        start=start,
        goal=goal,
    )

    st.write(explanation)


elif (
    not st.session_state.planning_message
    and not input_error
):

    st.info(
        "Adjust the mission inputs and click "
        "'Plan and Validate Route' to calculate a route."
    )


# ==================================================
# 13. AI AGENT'S MISSION INTERPRETATION
# ==================================================

# Display the AI interpretation only when:
# 1. An agent summary exists.
# 2. The selected planning mode is AI-Assisted Mode.
if (
    st.session_state.agent_summary
    and st.session_state.result_mode == AI_MODE
):

    # Create a collapsible section for the AI agent's explanation.
    with st.expander(
        "View AI Agent's Mission Interpretation",
        expanded=False,
    ):

        # Display the summary generated by the AI agent.
        st.write(st.session_state.agent_summary)

        # Explain which results have been independently verified.
        st.info(
            "Verification note: The route's move count and safety "
            "are calculated and checked by deterministic Python tools. "
            "The alternative-route search may be incomplete, so a "
            "globally minimum number of turns is not necessarily proven. "
            "Energy efficiency has not been measured using a flight "
            "energy model."
        )

        # Clarify the role of the AI agent.
        st.caption(
            "The AI interpretation is explanatory. The displayed "
            "route and safety status are checked separately by the "
            "deterministic planner and validator."
        )