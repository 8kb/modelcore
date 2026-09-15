"""
ArtifactStore: what ModelManager reads/writes model and optimizer artifacts through. Core owns
the artifact *format* (what a model/optimizer state looks like); a store just knows where bytes
live. FileSystemStore is a directory+step convention (model_{step:06d}.pt / meta_{step:06d}.json's
"model_config" key / optim_{step:06d}_rank{N}.pt) any host application can point at its own
checkpoint directory -- it hands ModelManager a FileSystemStore instead of doing the
torch.load/torch.save itself.

A store is deliberately narrow: read/write a model state dict, read/write an optimizer state dict
per rank, read/write the config dict. Anything else about a checkpoint (which tag, which step,
val_bpb, tokenizer fingerprint, dataloader state, ...) is naming/metadata policy that belongs to
whoever constructs the store, not to modelcore.
"""
import json
import os
import re

import torch


class ArtifactStore:
    """Protocol every store implements. Not an ABC -- duck typing is enough, and modelcore has no
    business enforcing what a caller's custom store subclasses from."""

    def read_config(self) -> dict:
        raise NotImplementedError

    def write_config(self, config_dict: dict) -> None:
        raise NotImplementedError

    def read_model_state(self, map_location=None) -> dict:
        raise NotImplementedError

    def write_model_state(self, state: dict) -> None:
        raise NotImplementedError

    def read_optimizer_state(self, rank: int = 0, map_location=None) -> dict | None:
        """Returns None if no optimizer state has been saved for this rank -- not every
        checkpoint has one (e.g. an RL checkpoint that never bothers)."""
        raise NotImplementedError

    def write_optimizer_state(self, state: dict, rank: int = 0) -> None:
        raise NotImplementedError


class FileSystemStore(ArtifactStore):
    """One checkpoint directory + step. write_config merges into meta_{step:06d}.json's
    "model_config" key rather than overwriting the file, since a host application typically writes
    its own sibling keys (val_bpb, user_config, tokenizer_fingerprint, ...) into the same file --
    each side only ever touches the key(s) it owns."""

    def __init__(self, checkpoint_dir: str, step: int):
        self.checkpoint_dir = checkpoint_dir
        self.step = step

    def _model_path(self) -> str:
        return os.path.join(self.checkpoint_dir, f"model_{self.step:06d}.pt")

    def _meta_path(self) -> str:
        return os.path.join(self.checkpoint_dir, f"meta_{self.step:06d}.json")

    def _optim_path(self, rank: int) -> str:
        return os.path.join(self.checkpoint_dir, f"optim_{self.step:06d}_rank{rank}.pt")

    def read_config(self) -> dict:
        with open(self._meta_path(), "r", encoding="utf-8") as f:
            return json.load(f)["model_config"]

    def write_config(self, config_dict: dict) -> None:
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        meta = {}
        if os.path.exists(self._meta_path()):
            with open(self._meta_path(), "r", encoding="utf-8") as f:
                meta = json.load(f)
        meta["model_config"] = config_dict
        with open(self._meta_path(), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    def read_model_state(self, map_location=None) -> dict:
        return torch.load(self._model_path(), map_location=map_location)

    def write_model_state(self, state: dict) -> None:
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        torch.save(state, self._model_path())

    def read_optimizer_state(self, rank: int = 0, map_location=None) -> dict | None:
        path = self._optim_path(rank)
        if not os.path.exists(path):
            return None
        return torch.load(path, map_location=map_location)

    def write_optimizer_state(self, state: dict, rank: int = 0) -> None:
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        torch.save(state, self._optim_path(rank))

    def read_meta(self) -> dict:
        """The full meta_{step:06d}.json dict (including "model_config") -- {} if it doesn't
        exist yet. A host's own sibling keys (val_bpb, user_config, tokenizer_fingerprint, ...)
        live in here alongside "model_config"; see update_meta for how a host adds/merges them."""
        if not os.path.exists(self._meta_path()):
            return {}
        with open(self._meta_path(), "r", encoding="utf-8") as f:
            return json.load(f)

    def update_meta(self, updates: dict) -> None:
        """Merges `updates` into meta_{step:06d}.json, creating it if needed -- the host-side
        mirror of write_config's own read-merge-write (see that method and the class docstring).
        Never touches the "model_config" key even if `updates` has one -- that key is write_config's
        alone, so a host merging its own sibling keys can't accidentally clobber it (or vice versa,
        write_config can't clobber a host's keys, since it only ever touches "model_config")."""
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        meta = self.read_meta()
        meta.update({k: v for k, v in updates.items() if k != "model_config"})
        with open(self._meta_path(), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)


def last_step(checkpoint_dir: str) -> int:
    """The highest step number among checkpoint_dir's model_<step>.pt files. Raises
    FileNotFoundError if there aren't any -- naming/tag policy (which directory, auto-discovery
    across tags) stays with the caller; this only knows the model_{step:06d}.pt naming convention
    FileSystemStore itself writes."""
    pattern = re.compile(r"model_(\d+)\.pt$")
    steps = []
    for filename in os.listdir(checkpoint_dir):
        match = pattern.search(filename)
        if match:
            steps.append(int(match.group(1)))
    if not steps:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    return max(steps)
