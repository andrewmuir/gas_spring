"""Find run configurations and simulation outputs with linked inputs."""

from dataclasses import dataclass, replace
import json
import os
from pathlib import Path

import pyarrow.parquet as parquet
import yaml


@dataclass(frozen=True)
class RunEntry:
    name: str
    input_path: Path
    output_path: Path
    config_path: Path | None = None


def run_name(settings, config_path):
    name = settings.get("run_name", Path(config_path).stem)
    if not isinstance(name, str) or not name.strip():
        raise ValueError("run_name must be a non-empty string")
    return name.strip()


def output_run_metadata(settings, config_path, input_path, output_path):
    """Paths in the saved output are relative to the output directory."""
    def link(path):
        path = Path(path).resolve()
        try:
            return os.path.relpath(path, output_path.parent)
        except ValueError:
            return str(path)

    return {
        "run_name": run_name(settings, config_path),
        "input_parquet": link(input_path),
        "run_yaml": link(config_path),
    }


def _linked_path(base, value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("run paths must be non-empty strings")
    return (base / value).resolve()


def _from_yaml(path):
    with path.open(encoding="utf-8-sig") as stream:
        settings = yaml.safe_load(stream)
    if not isinstance(settings, dict) or not {"input_parquet", "output_parquet"} <= settings.keys():
        return None
    return RunEntry(
        run_name(settings, path),
        _linked_path(path.parent, settings["input_parquet"]),
        _linked_path(path.parent, settings["output_parquet"]),
        path,
    )


def _from_output(path):
    footer = parquet.read_metadata(path).metadata or {}
    attrs = json.loads(footer.get(b"PANDAS_ATTRS", b"{}"))
    metadata = attrs.get("run_metadata") if isinstance(attrs, dict) else None
    if not isinstance(metadata, dict) or "input_parquet" not in metadata:
        return None
    config = metadata.get("run_yaml")
    return RunEntry(
        run_name(metadata, path),
        _linked_path(path.parent, metadata["input_parquet"]),
        path,
        _linked_path(path.parent, config) if config is not None else None,
    )


def scan_runs(directory):
    """Scan a directory tree without loading any Parquet data rows."""
    entries = {}
    results = {}
    for folder, subdirs, filenames in os.walk(directory):
        subdirs[:] = sorted(
            name for name in subdirs if not name.startswith(".") and name != "__pycache__"
        )
        for filename in sorted(filenames):
            if filename.startswith("."):
                continue
            candidate = Path(folder) / filename
            suffix = candidate.suffix.lower()
            if suffix not in (".yaml", ".yml", ".parquet", ".pq"):
                continue
            try:
                path = candidate.resolve()
                if suffix in (".yaml", ".yml"):
                    entry = _from_yaml(path)
                    if entry is not None:
                        entries.setdefault(entry.output_path, entry)
                elif suffix in (".parquet", ".pq"):
                    entry = _from_output(path)
                    if entry is not None:
                        results[path] = entry
            except (OSError, ValueError, yaml.YAMLError):
                continue
    for output_path, entry in entries.items():
        saved = results.get(output_path)
        if saved is None and output_path.is_file():
            try:
                saved = _from_output(output_path)
            except (OSError, ValueError):
                pass
        if saved is not None:
            results[output_path] = replace(entry, input_path=saved.input_path)
        else:
            results[output_path] = entry
    return sorted(results.values(), key=lambda entry: (entry.name.casefold(), str(entry.output_path)))
