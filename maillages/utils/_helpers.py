from __future__ import annotations

import ast


def deserialize_dict(data: dict) -> dict:
    """
    Recursively converts stringified tuples back to tuple keys.

    Parameters
    ----------
    data : dict
        Input dictionary with potentially stringified tuple keys.

    Returns
    -------
    dict
        Output dictionary with tuple keys restored.

    """
    if isinstance(data, dict):
        new_dict = {}

        for k, v in data.items():
            new_k = k

            if isinstance(k, str) and k.startswith("(") and k.endswith(")"):
                try:
                    parsed_key = ast.literal_eval(k)

                    if isinstance(parsed_key, tuple):
                        new_k = parsed_key

                except (ValueError, SyntaxError):
                    pass

            new_dict[new_k] = deserialize_dict(v)

        return new_dict

    else:
        return data


def serialize_dict(data: dict) -> dict:
    """
    Recursively converts tuple keys to strings in nested dicts/lists.

    Parameters
    ----------
    data : dict
        Input dictionary with potentially tuple keys.

    Returns
    -------
    dict
        Output dictionary with tuple keys converted to strings.

    """
    if isinstance(data, dict):
        return {
            str(k) if isinstance(k, tuple) else k: serialize_dict(v)
            for k, v in data.items()
        }

    else:
        return data
