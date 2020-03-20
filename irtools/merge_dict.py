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
