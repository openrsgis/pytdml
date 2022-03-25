# ------------------------------------------------------------------------------
#
# Project: pytdml
#
# ------------------------------------------------------------------------------
#
# ------------------------------------------------------------------------------


from typing import Iterable


def json_empty(item):
    """
    Check if a json item is empty
    Only null, empty collections and empty strings are considered "json empty"
    """
    if item is None:
        return True
    elif isinstance(item, Iterable):
        return not item
    elif isinstance(item, str):
        return item == ""
    else:
        return False


def remove_empty(item):
    """
    Remove empty items from a json item
    """
    if isinstance(item, dict):
        new_item = {k: remove_empty(v) for k, v in item.items()}
        return {k: v for k, v in new_item.items() if not json_empty(v)}
    elif isinstance(item, (list, tuple)):
        new_item = [remove_empty(v) for v in item]
        return [v for v in new_item if not json_empty(v)]
    else:
        return item
