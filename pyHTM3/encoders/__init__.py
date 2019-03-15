from .noop_encoder import NoopEncoder
from .sanity_encoder import SanityEncoder
from ..env.bandit import Bandit
from ..env.sanity import Sanity


def encoder_for_env(env, htmrl_config):
    t = type(env)
    if t == Bandit:
        return NoopEncoder(htmrl_config["input_size"])
    elif t == Sanity:
        return SanityEncoder(env.size)
    else:
        raise RuntimeError("No known encoder")