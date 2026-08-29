#!/usr/bin/env python3
import argparse
import copy

import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Create a WiSE-FT interpolated checkpoint.")
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--finetuned-checkpoint", required=True)
    parser.add_argument("--alpha", required=True, type=float, help="Interpolation weight for finetuned checkpoint.")
    parser.add_argument("--output-checkpoint", required=True)
    return parser.parse_args()


def get_state_dict(ckpt):
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        return ckpt["model_state_dict"]
    return ckpt


def interpolate_state_dict(base_state, finetuned_state, alpha):
    out = {}
    for key, base_tensor in base_state.items():
        if key not in finetuned_state:
            out[key] = base_tensor
            continue
        ft_tensor = finetuned_state[key]
        if torch.is_tensor(base_tensor) and torch.is_tensor(ft_tensor):
            if base_tensor.shape == ft_tensor.shape and torch.is_floating_point(base_tensor):
                out[key] = (1.0 - alpha) * base_tensor + alpha * ft_tensor
            else:
                out[key] = ft_tensor
        else:
            out[key] = ft_tensor
    for key, value in finetuned_state.items():
        if key not in out:
            out[key] = value
    return out


def main():
    args = parse_args()
    base_ckpt = torch.load(args.base_checkpoint, map_location="cpu")
    finetuned_ckpt = torch.load(args.finetuned_checkpoint, map_location="cpu")

    base_state = get_state_dict(base_ckpt)
    finetuned_state = get_state_dict(finetuned_ckpt)
    interpolated_state = interpolate_state_dict(base_state, finetuned_state, args.alpha)

    if isinstance(finetuned_ckpt, dict) and "model_state_dict" in finetuned_ckpt:
        output_ckpt = copy.deepcopy(finetuned_ckpt)
        output_ckpt["model_state_dict"] = interpolated_state
        output_ckpt["wise_ft"] = {
            "base_checkpoint": args.base_checkpoint,
            "finetuned_checkpoint": args.finetuned_checkpoint,
            "alpha": args.alpha,
        }
    else:
        output_ckpt = interpolated_state

    torch.save(output_ckpt, args.output_checkpoint)
    print(args.output_checkpoint)


if __name__ == "__main__":
    main()
