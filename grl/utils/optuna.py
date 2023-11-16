from typing import Any

def until_successful(func, *args, **kwargs) -> Any:
    """Repeatedly try calling function until successful.

    Returns:
        The first successful result
    """
    n_attempts = 0
    while True:
        n_attempts += 1
        if n_attempts >= 100 and n_attempts % 100 == 0:
            print(f'Failed to call function {n_attempts} in a row!?')
        try:
            result = func(*args, **kwargs)
        except RuntimeError:
            continue
        else:
            break
    return result
