"""Train the PPO altitude controller on validation-year missions.

The test year is never seen during training. The trained policy is then flown
by experiments/control_montecarlo.py on the same test missions as every other
controller.

Usage:  python experiments/train_rl.py --config configs/experiment.yaml [--steps 1000000]
"""
from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd

from stratoballoon import viz
from stratoballoon.atmosphere import Atmosphere
from stratoballoon.experiment import Context
from stratoballoon.forecasting.provider import FieldForecast
from stratoballoon.rl import make_vec_env
from stratoballoon.runlog import Run

log = logging.getLogger("rl")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--steps", type=int, default=1_000_000)
    ap.add_argument("--envs", type=int, default=64)
    a = ap.parse_args()
    ctx = Context(a.config)
    out = ctx.out / "rl"
    run = Run(out, {**ctx.cfg, "rl_steps": a.steps}, ctx.paths)
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback

    # Validation year, with a day of history; never the test year.
    vy = ctx.test_year - 1
    A = Atmosphere.from_files(ctx.paths, f"{vy - 1}-12-25", f"{vy}-12-31")
    best = ctx.best_model()
    t = pd.DatetimeIndex(A.times)
    issue = np.where((t.hour % 6 == 0) & (np.arange(len(t)) >= ctx.n_hist - 1))[0]
    prov = FieldForecast(A, ctx.model(best), issue, ctx.n_hist, ctx.horizons, best)
    starts = np.where((t >= pd.Timestamp(vy, 1, 2)) & (t <= pd.Timestamp(vy, 12, 27))
                      & (t.hour % 6 == 0))[0]
    env = make_vec_env(A, prov, ctx.conformal(best), ctx.params, ctx.band, starts,
                       n_envs=a.envs, duration_h=float(ctx.cfg["mission_hours"]), seed=ctx.seed)

    curve = []

    class Log(BaseCallback):
        def _on_rollout_end(self):
            r = self.model.rollout_buffer.rewards
            curve.append({"timesteps": self.num_timesteps, "mean_reward": float(r.mean())})

        def _on_step(self):
            return True

    model = PPO("MlpPolicy", env, n_steps=48, batch_size=768, n_epochs=8, gamma=0.97,
                learning_rate=3e-4, ent_coef=0.01, seed=ctx.seed, verbose=0,
                policy_kwargs={"net_arch": [128, 128]})
    model.learn(total_timesteps=a.steps, callback=Log())
    model.save(out / "ppo_policy")
    c = pd.DataFrame(curve)
    c.to_csv(out / "training_curve.csv", index=False)
    log.info("trained %d steps; final mean reward per decision %.3f", a.steps,
             c.mean_reward.iloc[-5:].mean())
    import matplotlib.pyplot as plt
    viz.apply_style()
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.plot(c.timesteps, c.mean_reward, color=viz.C1)
    ax.set_xlabel("Training decisions")
    ax.set_ylabel("Mean reward per decision")
    ax.set_title(f"PPO training on {vy} missions (validation year)")
    viz.finish(fig, out / "training_curve.png", viz.source_note("sim"))
    run.finish(best_model=best, train_year=vy)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
