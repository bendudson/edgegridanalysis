def to_yaml(d, indent = 0):
    """
    Convert a dictionary to a YAML string
    """
    result = ""
    for key, value in d.items():
        result += (" " * indent) + key + ":"
        if isinstance(value, dict):
            # A nested dictionary
            result += "\n" + to_yaml(value, indent = indent + 2)
        else:
            # A value
            result += " " + str(value) + "\n"
    return result

