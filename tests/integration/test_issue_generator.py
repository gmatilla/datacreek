from scripts.create_issues import parse_tasks

SAMPLE = """\
## Section A
* [ ] Task one
* [x] Done task
## Section B
* [ ] Another task
"""


def test_parse_tasks() -> None:
    tasks = parse_tasks(SAMPLE.splitlines())
    assert tasks == [  # noqa: S101
        ("Section A", "Task one"),
        ("Section B", "Another task"),
    ]
