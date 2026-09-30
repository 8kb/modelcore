"""
Checkpoint converter, modelcore.v1/v2 -> v3. The config upgrader (modelcore.config.upgrade) is pure
dict surgery and never sees weights; this is the separate, explicit half that moves a checkpoint's
weights to the v3 layout (attn -> mixer, mlp -> ffn, the gate / value embedding / lambdas under
`features`) and writes the upgraded config beside them. ModelManager.load_model refuses a pre-v3
checkpoint and points here.

Optimizer state is NOT converted: v3 parameters are reached in a different order, and optimizer
state is positional, so a converted checkpoint cannot resume training -- it loads for eval/serve/
fine-tune-from-scratch-optimizer. Host metadata sitting beside the config (a FileSystemStore's
meta_*.json siblings) is carried over.

    python -m modelcore.convert SRC_DIR DST_DIR [--step N]
"""
import argparse

from modelcore.config.spec import FORMAT
from modelcore.config.upgrade import remap_v2_name, upgrade_to_v3
from modelcore.store import FileSystemStore, last_step


def convert_state_v2_to_v3(state: dict) -> dict:
    """Rename every key of a v2 model state dict to its v3 spelling, preserving order. Raises if two
    keys collide after renaming."""
    out = {}
    for key, value in state.items():
        new_key = remap_v2_name(key)
        if new_key in out:
            raise ValueError(f"state dict key collision after conversion: {new_key!r}")
        out[new_key] = value
    return out


def convert_checkpoint_v2_to_v3(src_store, dst_store) -> dict:
    """Read src_store's config and weights, write their v3 form to dst_store; returns the v3 config
    dict. src and dst must be different locations. A checkpoint already at v3 is copied through."""
    config = upgrade_to_v3(src_store.read_config())
    assert config["format"] == FORMAT
    state = src_store.read_model_state(map_location="cpu")
    state = convert_state_v2_to_v3(state)  # v3 keys have no rule, so a v3 state passes through unchanged
    dst_store.write_config(config)
    dst_store.write_model_state(state)
    if hasattr(src_store, "read_meta") and hasattr(dst_store, "update_meta"):
        dst_store.update_meta(src_store.read_meta())
    return config


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m modelcore.convert", description=__doc__.split("\n\n")[0])
    parser.add_argument("src_dir")
    parser.add_argument("dst_dir")
    parser.add_argument("--step", type=int, default=None, help="checkpoint step (default: the last one in SRC_DIR)")
    args = parser.parse_args(argv)
    step = last_step(args.src_dir) if args.step is None else args.step
    convert_checkpoint_v2_to_v3(FileSystemStore(args.src_dir, step), FileSystemStore(args.dst_dir, step))
    print(f"converted {args.src_dir} step {step} -> {args.dst_dir} (optimizer state not converted)")


if __name__ == "__main__":
    main()
