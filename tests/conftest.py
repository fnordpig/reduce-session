import json
import pytest


@pytest.fixture
def sample_session(tmp_path):
    """Create a minimal but realistic session JSONL for testing."""
    messages = [
        {
            "type": "system",
            "uuid": "sys-1",
            "message": {"content": "You are Claude."},
            "timestamp": "2026-03-23T01:00:00Z",
        },
        {
            "type": "user",
            "uuid": "u-1",
            "parentUuid": "sys-1",
            "message": {"content": "Hello"},
            "timestamp": "2026-03-23T01:01:00Z",
        },
        {
            "type": "assistant",
            "uuid": "a-1",
            "parentUuid": "u-1",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "Hi there!"}],
                "usage": {
                    "input_tokens": 10,
                    "cache_read_input_tokens": 100,
                    "cache_creation_input_tokens": 5,
                },
            },
            "timestamp": "2026-03-23T01:01:30Z",
        },
        {
            "type": "progress",
            "uuid": "p-1",
            "parentUuid": "a-1",
            "data": {"type": "hook_progress"},
            "timestamp": "2026-03-23T01:01:31Z",
        },
        {
            "type": "user",
            "uuid": "u-2",
            "parentUuid": "p-1",
            "message": {
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tu-1",
                        "content": "file contents here " * 500,
                    }
                ]
            },
            "timestamp": "2026-03-23T01:02:00Z",
        },
        {
            "type": "assistant",
            "uuid": "a-2",
            "parentUuid": "u-2",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "I see the file."}],
                "usage": {
                    "input_tokens": 20,
                    "cache_read_input_tokens": 200,
                    "cache_creation_input_tokens": 10,
                },
            },
            "timestamp": "2026-03-23T01:02:30Z",
        },
    ]
    path = tmp_path / "test-session.jsonl"
    with open(path, "w") as f:
        for msg in messages:
            f.write(json.dumps(msg) + "\n")
    return path


@pytest.fixture
def sample_project_dir(tmp_path, sample_session):
    project = tmp_path / "projects" / "-Users-test-src-myproject"
    project.mkdir(parents=True)
    import shutil

    dest = project / "abc12345-dead-beef-cafe-123456789abc.jsonl"
    shutil.copy(sample_session, dest)
    return project
