"""
Env loader and actuator introspection helpers for flybody-based MuJoCo tasks.
Assumes dm_control-style API: env.reset(), env.step(action), env.action_spec(), env.physics.render().
"""

from typing import List
import importlib

def load_env(mode: str = "walk", **kwargs):
    """
    Load a flybody environment by mode.
    mode: "walk", "walk_on_ball", "flight", or your custom task key.
    kwargs: forwarded to the flybody env constructor or task factory.
    """
    try:
        fb = importlib.import_module("flybody.fly_envs")
    except ModuleNotFoundError as e:
        raise RuntimeError("Could not import flybody.fly_envs. Ensure flybody is on PYTHONPATH.") from e

    if mode.lower() in ("walk", "walking", "walk_imitation"):
        env = fb.WalkImitation(**kwargs)
    elif mode.lower() in ("walk_on_ball", "ball"):
        env = fb.WalkOnBall(**kwargs)
    elif mode.lower() in ("flight", "fly", "flight_wpg"):
        env = fb.FlightImitationWBPG(**kwargs)
    else:
        if hasattr(fb, "make_env"):
            env = fb.make_env(mode, **kwargs)
        else:
            raise ValueError(f"Unknown mode '{mode}'. Update env_loader.load_env.")
    return env

def list_actuator_names(env) -> List[str]:
    names = []
    ph = env.physics
    try:
        nu = ph.model.nu
        for i in range(nu):
            names.append(ph.model.id2name(i, "actuator"))
    except Exception:
        try:
            names = list(ph.named.model.actuator._names)
        except Exception as e:
            raise RuntimeError("Unable to introspect actuator names from env.physics.") from e
    return names

def actuator_index_map(env):
    names = list_actuator_names(env)
    return {n: i for i, n in enumerate(names)}

def control_dt_seconds(env) -> float:
    task = getattr(env, "task", None)
    if task is not None:
        for key in ("dt_control_s", "dt_control_ms", "control_dt_s", "control_dt_ms"):
            if hasattr(task, key):
                v = getattr(task, key)
                return float(v) * 1e-3 if "ms" in key else float(v)
    physics_dt = getattr(env.physics.model, "opt", None)
    mj_timestep = getattr(physics_dt, "timestep", 0.0002)
    return 4.0 * mj_timestep
