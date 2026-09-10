"""Shared helpers for serialized check payloads (YAML/JSON)."""


def checks_dict_to_list(checks):
    """Convert legacy dict-shaped dataframe checks to list form.

    Older formats stored checks as ``{check_name: stats}``; newer formats use
    a list of dicts with ``options.check_name`` set.
    """
    if checks is None or not isinstance(checks, dict):
        return checks
    out = []
    for check_name, check in checks.items():
        if not isinstance(check, dict):
            check = {"value": check}
        if "options" not in check:
            check["options"] = {}
        check["options"]["check_name"] = check_name
        out.append(check)
    return out
