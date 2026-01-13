# Runtime Issues and Fixes Log

## Issue 1: Agent Unresponsive After Generating Travel Plan

### Symptom
After the agent successfully generated the initial travel plan, any subsequent messages sent by the user were ignored. The backend agent continued to run, but no response was displayed on the Streamlit UI.

### Root Cause
The LangGraph workflow definition in `agent_graph.py` was configured to terminate the conversation graph after the planning phase.
- The edge from `create_final_plan` pointed to the `END` node.
- This meant that once a plan was created, the state machine reached a terminal state and could not accept further inputs or resume execution for that conversation thread.

### Solution
Modified the graph structure in `agent_graph.py` to enable a continuous conversation loop.
- **Change**: Redirected the edge from `create_final_plan` to `get_next_user_message`.
- **Effect**: After generating a plan, the agent now waits for the next user input, allowing for follow-up questions or adjustments to the plan.

```python
# Before
graph.add_edge("create_final_plan", END)

# After
graph.add_edge("create_final_plan", "get_next_user_message")
```

## Issue 2: Shared Conversation State Across Sessions

### Symptom
The `thread_id` used to identify unique conversation threads in LangGraph was defined globally in the `streamlit_ui.py` module.

### Root Cause
Streamlit re-runs the script for interactions but module-level global variables can persist or be shared in unexpected ways depending on the runner context, and more importantly, the logic didn't strictly bind the thread ID to the specific user session state effectively for Graph invocation.
- `thread_id` was initialized at the module level: `thread_id = get_thread_id()`.

### Solution
Updated `streamlit_ui.py` to strictly use Streamlit's `session_state` for managing the `thread_id`.
- **Change**: Removed the global `thread_id` variable and the `get_thread_id` function.
- **Change**: Updated the `invoke_agent_graph` function to retrieve `thread_id` directly from `st.session_state.thread_id`.

```python
# Before
config = {
    "configurable": {
        "thread_id": thread_id
    }
}

# After
config = {
    "configurable": {
        "thread_id": st.session_state.thread_id
    }
}
```

These changes ensure that:
1. The conversation does not end prematurely.
2. Each user session has a unique and isolated conversation thread.
