from typing import Any


def pytest_benchmark_update_machine_info(config, machine_info: dict[str, Any]) -> None:  # noqa: ARG001
    # a bit annoying, hostname can be volatile, e.g. when running under docker
    machine_info["node"] = "redacted"
