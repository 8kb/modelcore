"""
Model: the one model class every modelcore.ModelConfig builds, driven entirely by the config
tree instead of hardcoded per-architecture assembly -- see modelcore/docs/architecture.md's
"The materialized config tree" section.

Model deliberately carries no accounting or optimizer methods (estimate_flops, kv_cache_spec,
setup_optimizer, ...): those need a model only to determine shapes/roles, which
ModelManager.stats()/create_optimizer() do from the outside, over layer_specs()/param_roles --
see modelcore/stats.py and modelcore/manager.py. Model's own surface is deliberately small:
build it, init its weights, run it forward.
"""
import torch
import torch.nn as nn

from modelcore.catalog import build_component
from modelcore.runtime import DEFAULT_RUNTIME


class Model(nn.Module):
    def __init__(self, config, runtime=None):
        """NOTE: may run under torch.device("meta") -- see modelcore/docs/architecture.md's
        "The meta-device footgun"."""
        super().__init__()
        self.config = config
        self.runtime = runtime or DEFAULT_RUNTIME
        padded_vocab_size = config.padded_vocab_size
        if padded_vocab_size != config.vocab_size:
            self.runtime.log(f"Padding vocab_size from {config.vocab_size} to {padded_vocab_size} for efficiency")

        # Build context: derived globals every component may ask to have injected (see each
        # catalog entry's `needs`), plus `shared` components (e.g. rope) built once here and
        # injected into whatever asks for them by name -- see modelcore.catalog.
        ctx = {
            "n_embd": config.n_embd, "vocab_size": config.vocab_size,
            "padded_vocab_size": padded_vocab_size, "sequence_len": config.sequence_len,
            "runtime": self.runtime,
        }
        shared = {}
        for name, spec in config.shared.items():
            shared[name] = build_component(spec, ctx)
            ctx[name] = shared[name]
        self.shared = nn.ModuleDict(shared)

        self.embedding = build_component(config.input, ctx)
        self.body = build_component(config.body, ctx)
        self.unembedding = build_component(config.output, ctx)

    @torch.no_grad()
    def init_weights(self):
        self.embedding.init_weights()
        for m in self.shared.values():
            m.init_weights()
        self.body.init_weights()
        self.unembedding.init_weights()

    def layer_specs(self):
        """Internal to modelcore: used by ModelManager.stats()/new_kv_cache(), not part of
        Model's own public contract (see the module docstring)."""
        return self.body.layer_specs()

    def get_device(self):
        return next(self.parameters()).device

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean', doc_args=None):
        """doc_args: optional modelcore.kernels.flash_attn.DocArgs restricting attention to within
        each packed row's own document (see build_doc_args) -- training only; always None during
        inference (kv_cache is not None), since one KV-cache row is one document at decode time."""
        x = self.embedding(idx, kv_cache)
        x = self.body(x, idx, kv_cache, doc_args)
        if kv_cache is not None:
            kv_cache.advance(idx.size(1))
        return self.unembedding(x, targets, loss_reduction)
