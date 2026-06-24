"""
Cosmos Policy inference server for xArm7.

Loads the model once at startup and exposes a single POST /act endpoint.
Mirrors the structure of cosmos_policy.experiments.robot.aloha.deploy but
trimmed to what new_cosmos_xarm_inference.py needs (no planning / value /
best-of-N). Reuses the same PolicyEvalConfig as the inference script so the
client and server stay in sync.

Request payload (JSON):
    {
        "primary_image": "<base64-encoded JPEG bytes>",
        "wrist_image":   "<base64-encoded JPEG bytes>",
        "proprio":       [j0, j1, j2, j3, j4, j5, j6],
        "task":          "put the carrot in the bowl"
    }

Response (JSON):
    {
        "actions": [[...7 floats...], ...]   # one entry per chunk timestep
    }

Usage example:
    python cosmos_policy_server.py \
        --ckpt_path /path/to/model.pt \
        --dataset_stats_path /path/to/dataset_statistics.json \
        --t5_text_embeddings_path /path/to/t5_embeddings.pkl \
        --host 0.0.0.0 --port 8777
"""

import argparse
import base64
import logging
import os
import traceback
from typing import Any, Dict

from cosmos_policy.experiments.robot.libero import new_cosmos_xarm_inference
import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from cosmos_policy.experiments.robot.cosmos_utils import (
    get_action,
    get_model,
    init_t5_text_embeddings_cache,
    load_dataset_stats,
)
from cosmos_policy.utils.utils import set_seed_everywhere

from cosmos_policy.experiments.robot.libero.new_cosmos_xarm_inference  import PolicyEvalConfig


def _decode_jpeg_b64(b64_str: str) -> np.ndarray:
    raw = base64.b64decode(b64_str)
    arr = np.frombuffer(raw, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Failed to decode JPEG image from base64 payload")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _to_serializable_actions(actions):
    """get_action returns a list of 1D numpy arrays. Convert to plain lists."""
    out = []
    for a in actions:
        if isinstance(a, np.ndarray):
            out.append(a.astype(float).tolist())
        else:
            out.append([float(x) for x in a])
    return out


class PolicyServer:
    def __init__(self, cfg: PolicyEvalConfig):
        self.cfg = cfg

        if cfg.deterministic:
            os.environ["DETERMINISTIC"] = "True"
        set_seed_everywhere(cfg.seed)

        init_t5_text_embeddings_cache(cfg.t5_text_embeddings_path)
        self.dataset_stats = load_dataset_stats(cfg.dataset_stats_path)
        self.model, _ = get_model(cfg)

    def get_server_action(self, payload: Dict[str, Any]):
        try:
            primary = _decode_jpeg_b64(payload["primary_image"])
            wrist = _decode_jpeg_b64(payload["wrist_image"])
            proprio = np.asarray(payload["proprio"], dtype=np.float32)
            task = payload["task"]

            obs = {
                "primary_image": primary,
                "wrist_image": wrist,
                "proprio": proprio,
            }

            out = get_action(
                self.cfg,
                self.model,
                self.dataset_stats,
                obs,
                task,
                seed=self.cfg.seed,
                randomize_seed=self.cfg.randomize_seed,
                num_denoising_steps_action=self.cfg.num_denoising_steps_action,
                generate_future_state_and_value_in_parallel=not (
                    self.cfg.ar_future_prediction
                    or self.cfg.ar_value_prediction
                    or self.cfg.ar_qvalue_prediction
                ),
            )

            return JSONResponse({"actions": _to_serializable_actions(out["actions"])})
        except Exception:
            logging.error(traceback.format_exc())
            return JSONResponse(status_code=500, content={"error": traceback.format_exc()})

    def run(self, host: str, port: int):
        self.app = FastAPI()
        self.app.post("/act")(self.get_server_action)
        uvicorn.run(self.app, host=host, port=port)


def _build_parser() -> argparse.ArgumentParser:
    deploy_dir = os.path.dirname(os.path.abspath(__file__))
    default_ckpt = "/workspace/cosmos_predict2_2b_480p_real_world_xarm_put_carrot_in_bowl_5hz/checkpoints/iter_000009000/model.pt"
    default_stats = "/workspace/cosmos_predict2_2b_480p_real_world_xarm_put_carrot_in_bowl_5hz/dataset_statistics.json"
    default_t5 ="/workspace/cosmos_predict2_2b_480p_real_world_xarm_put_carrot_in_bowl_5hz/checkpoints/t5_embeddings.pkl"

    p = argparse.ArgumentParser()
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8777)

    p.add_argument("--ckpt_path", default=default_ckpt)
    p.add_argument("--dataset_stats_path", default=default_stats)
    p.add_argument("--t5_text_embeddings_path", default=default_t5)

    p.add_argument("--config", default="cosmos_predict2_2b_480p_libero__inference_only")
    p.add_argument("--config_file", default="cosmos_policy/config/config.py" )
    p.add_argument("--suite", default=None)

    p.add_argument("--chunk_size", type=int, default=16)
    p.add_argument("--num_open_loop_steps", type=int, default=16)
    p.add_argument("--num_denoising_steps_action", type=int, default=None)
    p.add_argument("--seed", type=int, default=20)
    return p


def _apply_overrides(cfg: PolicyEvalConfig, args: argparse.Namespace) -> PolicyEvalConfig:
    for field in (
        "ckpt_path",
        "dataset_stats_path",
        "t5_text_embeddings_path",
        "config",
        "config_file",
        "suite",
        "chunk_size",
        "num_open_loop_steps",
        "num_denoising_steps_action",
        "seed",
    ):
        val = getattr(args, field, None)
        if val is not None:
            setattr(cfg, field, val)
    return cfg


def main():
    args = _build_parser().parse_args()
    cfg = PolicyEvalConfig()
    # cfg = _apply_overrides(cfg, args)
    print(cfg)

    server = PolicyServer(cfg)
    server.run(args.host, args.port)


if __name__ == "__main__":
    main()
