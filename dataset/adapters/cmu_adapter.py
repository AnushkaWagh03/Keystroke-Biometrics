import os
import re

from dataset.base_dataset import BaseKeystrokeDataset


def _get_dummy_data():
    """Return dummy data when real CMU path is missing or empty."""
    return [
        {
            "user_id": "s1",
            "key_timings": [
                {"key": "h", "hold_time": 85},
                {"key": "e", "hold_time": 72},
                {"key": "l", "hold_time": 90},
                {"key": "l", "hold_time": 88},
                {"key": "o", "hold_time": 95},
            ],
        },
        {
            "user_id": "s2",
            "key_timings": [
                {"key": "t", "hold_time": 92},
                {"key": "e", "hold_time": 78},
                {"key": "s", "hold_time": 81},
                {"key": "t", "hold_time": 89},
            ],
        },
    ]


def _parse_cmu_line(line):
    """
    Parse a single line in CMU format: hold_time and key (whitespace-separated).
    Returns (hold_time, key) or None if line is invalid.
    """
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    parts = line.split()
    if len(parts) < 2:
        return None

    try:
        hold = float(parts[0])
        key = parts[-1]
        if key and hold >= 0:
            return hold, key
    except (ValueError, IndexError):
        return None

    return None


def _load_user_file(filepath):
    """Load one .txt file into a list of key_timings dicts."""
    key_timings = []
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                parsed = _parse_cmu_line(line)
                if parsed:
                    hold_time, key = parsed
                    key_timings.append({"key": key, "hold_time": hold_time})
    except OSError:
        return []

    return key_timings


def _find_user_folders(root_path):
    """Find folders named s followed by digits (s1, s2, s01, s002, etc.)."""
    if not os.path.isdir(root_path):
        return []

    pattern = re.compile(r"^s\\d+$", re.IGNORECASE)
    folders = []

    for name in os.listdir(root_path):
        path = os.path.join(root_path, name)
        if os.path.isdir(path) and pattern.match(name):
            folders.append((name, path))

    return sorted(folders, key=lambda x: x[0].lower())


def _load_cmu_from_path(dataset_path):
    """
    Load real CMU data from dataset_path.
    Expects user folders (s1, s2, ...) each containing .txt files.
    Returns list of samples: [{"user_id": "s1", "key_timings": [...]}, ...].
    """
    data = []

    for user_id, user_path in _find_user_folders(dataset_path):
        for filename in sorted(os.listdir(user_path)):
            if not filename.lower().endswith(".txt"):
                continue
            filepath = os.path.join(user_path, filename)
            if not os.path.isfile(filepath):
                continue

            key_timings = _load_user_file(filepath)
            if key_timings:
                data.append({"user_id": user_id, "key_timings": key_timings})

    return data


class CMUDataset(BaseKeystrokeDataset):
    def __init__(self, config):
        super().__init__(config)
        self._dataset_path = None

    def _resolve_dataset_path(self):
        """Resolve dataset path from config with default ./data/cmu/."""
        if self._dataset_path is not None:
            return self._dataset_path

        dataset_config = self.config.get("dataset", {})
        path = dataset_config.get("path", "./data/cmu/")

        if not os.path.isabs(path):
            path = os.path.normpath(os.path.join(os.getcwd(), path))

        self._dataset_path = path
        return self._dataset_path

    def load_data(self):
        path = self._resolve_dataset_path()

        if not os.path.exists(path):
            print(f"CMU dataset path not found: {path}, using dummy data")
            self.data = _get_dummy_data()
            return self

        loaded = _load_cmu_from_path(path)
        if not loaded:
            print(f"No CMU data found under {path}, using dummy data")
            self.data = _get_dummy_data()
        else:
            print(f"Loaded CMU dataset from {path}: {len(loaded)} samples")
            self.data = loaded

        return self

    def normalize_to_standard_format(self):
        """Data is already in standard format with key_timings list."""
        return self.data
