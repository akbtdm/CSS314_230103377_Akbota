"""Shared helpers: result paths and loaders for the salted parameters and host profile."""
import json
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")


def results_path(name: str) -> str:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    return os.path.join(RESULTS_DIR, name)


def load_json(name: str):
    path = os.path.join(RESULTS_DIR, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Run `python3 src/profile_env.py --student-id <ID>` first.")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def load_salted_params():
    return load_json("salted_params.json")


def load_host_profile():
    return load_json("host_profile.json")


def cache_line_bytes(default: int = 64) -> int:
    try:
        value = load_host_profile()["caches"]["cache_line_bytes"]
        return int(value) if value else default
    except (FileNotFoundError, KeyError, TypeError, ValueError):
        return default
