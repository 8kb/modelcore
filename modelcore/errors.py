"""
Config validation result types. validate_config() never raises and never stops at the first
problem -- it walks the whole tree and returns every error it finds, each anchored to a path a
human can find in the config JSON (e.g. "body.blocks[3].n_kv_head", "shared.rope.head_dim").
"""
from dataclasses import dataclass, field


@dataclass
class ConfigError:
    path: str
    message: str

    def __str__(self):
        return f"{self.path}: {self.message}" if self.path else self.message


@dataclass
class ValidationReport:
    errors: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def __bool__(self):
        return self.ok

    def __str__(self):
        if self.ok:
            return "valid"
        return "\n".join(str(e) for e in self.errors)
