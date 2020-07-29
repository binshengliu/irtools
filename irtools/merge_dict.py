from typing import Any, Dict, List, Optional


def merge_dict_of_dict(list_of_dicts):
    result = {}
    for one in list_of_dicts:
        for k, v in one.items():
            result.setdefault(k, {}).update(v)
    return result


def merge_dict_of_list(list_of_dicts):
    result = {}
    for one in list_of_dicts:
        for k, v in one.items():
            result[k] = result.get(k, []) + v
    return result


_T = Dict[Any, Any]


# https://stackoverflow.com/a/7205107/955952
def merge(
    a: Dict[Any, Any], b: Dict[Any, Any], path: Optional[List[str]] = None
) -> Dict[Any, Any]:
    "merges b into a"
    if path is None:
        path = []
    for key in b:
        if key in a:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                merge(a[key], b[key], path + [str(key)])
            elif a[key] == b[key]:
                pass  # same leaf value
            else:
                raise Exception("Conflict at %s" % ".".join(path + [str(key)]))
        else:
            a[key] = b[key]
    return a
