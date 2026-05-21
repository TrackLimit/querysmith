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
