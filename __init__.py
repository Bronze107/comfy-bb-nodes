import random


def load_wildcard_lines(path):
    with open(path, encoding="utf-8-sig") as f:
        lines = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]
    if not lines:
        raise ValueError(f"No valid entries in wildcard file: {path}")
    return lines


def pick_random_line(lines, seed):
    if not lines:
        raise ValueError("Cannot pick from an empty list")
    return random.Random(seed).choice(lines)
