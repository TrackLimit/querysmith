from agent.messages import MessageManager


def test_message_manager_basic_flow():
    manager = MessageManager(system="You are a helpful assistant.")
    manager.add_user_message("Hello, Claude")
    manager.add_assistant_message("Hello!")
    manager.add_user_message("Can you describe LLMs to me?")

    kwargs = manager.to_anthropic_kwargs()
    assert kwargs["system"] == "You are a helpful assistant."
    assert kwargs["messages"] == [
        {"role": "user", "content": "Hello, Claude"},
        {"role": "assistant", "content": "Hello!"},
        {"role": "user", "content": "Can you describe LLMs to me?"},
    ]


def test_message_manager_carries_tool_messages():
    manager = MessageManager()
    manager.add_assistant_message(
        [
            {
                "type": "tool_use",
                "id": "t1",
                "name": "execute_sql",
                "input": {"query": "SELECT 1"},
            }
        ]
    )
    manager.add_user_message(
        [{"type": "tool_result", "tool_use_id": "t1", "content": "[]"}]
    )
    messages = manager.to_anthropic_kwargs()["messages"]
    assert messages[0]["content"][0]["type"] == "tool_use"
    assert messages[1]["content"][0]["type"] == "tool_result"
