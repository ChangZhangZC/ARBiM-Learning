"""Shared successful-expert episode selection; no success inference."""
import re

def resolve_episode_ids(selection, total_episodes):
    if type(total_episodes) is not int or total_episodes < 1:
        raise ValueError("Expected a positive dataset episode count")
    if selection == "all":
        return list(range(total_episodes))
    if isinstance(selection, str):
        match = re.fullmatch(r"([0-9]+):([0-9]+)", selection)
        if not match:
            raise ValueError('episode_ids must be a list, "start:stop", or all')
        start, stop = map(int, match.groups())
        if not 0 <= start < stop <= total_episodes:
            raise ValueError("Episode range outside dataset or empty")
        return list(range(start, stop))
    if not isinstance(selection, list) or not selection:
        raise ValueError("Explicit nonempty successful episode_ids required")
    if any(type(i) is not int or not 0 <= i < total_episodes for i in selection):
        raise ValueError("Episode ID must be an integer within dataset")
    if len(set(selection)) != len(selection):
        raise ValueError("Duplicate episode IDs")
    return list(selection)
