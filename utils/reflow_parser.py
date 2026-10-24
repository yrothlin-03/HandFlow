import argparse
from copy import deepcopy

import yaml


def _flatten(config, prefix=""):
    flattened = {}
    for key, value in config.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flattened.update(_flatten(value, path))
        else:
            flattened[path] = value
    return flattened


def _boolean(value):
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes", "y", "on"}:
        return True
    if normalized in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean: {value}")


def _convert(value, reference, path):
    if value.strip().lower() in {"none", "null", "~"}:
        return None
    try:
        if isinstance(reference, bool):
            return _boolean(value)
        if isinstance(reference, int):
            return int(value)
        if isinstance(reference, float):
            return float(value)
        if isinstance(reference, str):
            return value
        parsed = yaml.safe_load(value)
        if reference is None:
            return parsed
        if not isinstance(parsed, type(reference)):
            raise TypeError
        return parsed
    except (TypeError, ValueError) as error:
        expected = type(reference).__name__
        raise argparse.ArgumentTypeError(
            f"Invalid value for {path}: {value!r}, expected type: {expected}"
        ) from error


def _set(config, path, value):
    target = config
    parts = path.split(".")
    for part in parts[:-1]:
        target = target[part]
    target[parts[-1]] = value


def parse_reflow_config(config, argv=None):
    flattened = _flatten(config)
    leaf_counts = {}
    for path in flattened:
        leaf = path.rsplit(".", 1)[-1]
        leaf_counts[leaf] = leaf_counts.get(leaf, 0) + 1

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--parsing",
        nargs="?",
        const=True,
        default=False,
        type=_boolean,
    )

    destinations = {}
    for path in flattened:
        leaf = path.rsplit(".", 1)[-1]
        options = [f"--{path}"]
        if leaf_counts[leaf] == 1 and leaf != path:
            options.append(f"--{leaf}")
        destination = f"override_{path.replace('.', '__')}"
        destinations[destination] = path
        parser.add_argument(
            *options,
            dest=destination,
            default=argparse.SUPPRESS,
            metavar="VALUE",
        )

    arguments = vars(parser.parse_args(argv))
    parsing = arguments.pop("parsing")
    if arguments and not parsing:
        parser.error("Overrides require --parsing true")

    updated = deepcopy(config)
    if parsing:
        for destination, raw_value in arguments.items():
            path = destinations[destination]
            value = _convert(raw_value, flattened[path], path)
            _set(updated, path, value)
    return updated
