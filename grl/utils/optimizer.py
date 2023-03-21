import optax

def get_optimizer(optimizer: str = 'sgd', step_size: float = 1e-3) -> optax.GradientTransformation:
    if optimizer == "sgd":
        return optax.sgd(step_size)
    # TODO: add kwargs or something for more fine-tuned optimization
    elif optimizer == "adam":
        return optax.adam(step_size)
    elif optimizer == 'rmsprop':
        return optax.rmsprop(step_size)
    else:
        raise NotImplementedError
