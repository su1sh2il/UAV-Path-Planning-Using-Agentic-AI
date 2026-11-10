
# Import asyncio to run asynchronous Python operations.
import asyncio

# Import Agent from Microsoft Agent Framework.
from agent_framework import Agent

# Import the Ollama chat client.
from agent_framework.ollama import OllamaChatClient


# Define the asynchronous main function.
async def main():

    # Confirm that the program entered the main function.
    print("STEP 1: Starting UAV Mission Planner", flush=True)

    # Connect to the locally installed Qwen3 4B model.
    client = OllamaChatClient(
        model="qwen3:4b"
    )
    print("STEP 2: Ollama client created", flush=True)

    # Create the Mission Planner Agent.
    agent = Agent(
        client=client,
        name="UAVMissionPlanner",
        instructions=(
            "You are a UAV Mission Planner Agent. "
            "Prepare a structured mission plan for a 2D grid. "
            "Identify the start, goal, obstacles, and movement rules. "
            "Do not claim a route is safe without validation."
        ),
    )
    print("STEP 3: Agent created", flush=True)

    # Define a small test mission.
    mission = (
        "Start: (0, 0). Goal: (9, 7). "
        "Obstacles: (2, 0), (2, 1), (2, 2). "
        "Grid: 10 by 10. Movement: four directions. "
        "Summarize these mission inputs."
    )

    # Send the mission to the agent and wait for its response.
    print("STEP 4: Sending mission to agent...", flush=True)
    result = await agent.run(mission)

    # Print the generated response text.
    print("STEP 5: Agent response received", flush=True)
    print(result.text, flush=True)


# Run the asynchronous main function.
if __name__ == "__main__":
    try:
        asyncio.run(main())

    # Display errors rather than silently hiding them.
    except Exception as error:
        print(f"ERROR: {type(error).__name__}: {error}", flush=True)
        raise
