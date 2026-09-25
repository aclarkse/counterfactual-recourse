"""Validation and indexing for partially ordered mediator blocks."""

from dataclasses import dataclass


@dataclass(frozen=True)
class MediatorSchema:
    layers: tuple[tuple[str, ...], ...]
    discrete_names: tuple[str, ...]
    continuous_names: tuple[str, ...]

    @classmethod
    def from_sfm_config(cls, sfm_cfg):
        discrete = sfm_cfg["mediators_disc"]
        continuous = sfm_cfg["mediators_cont"]
        layers = sfm_cfg.get("mediator_layers")
        if layers is None:
            # Compatibility for checkpoints trained before explicit block
            # schemas were saved.
            layers = [[name] for name in discrete]
            if continuous:
                layers.append(list(continuous))
        return cls.build(layers, discrete, continuous)

    @classmethod
    def build(cls, layers, discrete_names, continuous_names):
        layers = tuple(tuple(block) for block in layers)
        discrete_names = tuple(discrete_names)
        continuous_names = tuple(continuous_names)
        expected = list(discrete_names + continuous_names)
        flattened = [name for block in layers for name in block]
        if len(flattened) != len(set(flattened)):
            raise ValueError("Each mediator must occur in exactly one block.")
        if set(flattened) != set(expected):
            missing = sorted(set(expected) - set(flattened))
            extra = sorted(set(flattened) - set(expected))
            raise ValueError(f"Invalid mediator layers; missing={missing}, extra={extra}")

        kind = {name: "discrete" for name in discrete_names}
        kind.update({name: "continuous" for name in continuous_names})
        for block in layers:
            block_kinds = {kind[name] for name in block}
            if len(block_kinds) != 1:
                raise ValueError(
                    f"Mixed-type block {block} is unsupported; use "
                    "type-homogeneous blocks."
                )

        continuous_layers = [i for i, block in enumerate(layers)
                             if block and kind[block[0]] == "continuous"]
        if len(continuous_layers) > 1:
            raise ValueError(
                "Only one joint continuous block is currently supported."
            )
        if continuous_layers and continuous_layers[0] != len(layers) - 1:
            raise ValueError("The joint continuous block must be the final block.")
        for block in layers:
            if block and kind[block[0]] == "discrete" and len(block) > 1:
                raise ValueError(
                    "Same-level categorical mediators require a joint categorical "
                    "model; only singleton categorical blocks are supported."
                )
        discrete_layer_indices = [
            next(i for i, block in enumerate(layers) if name in block)
            for name in discrete_names
        ]
        if discrete_layer_indices != sorted(discrete_layer_indices):
            raise ValueError(
                "mediators_disc must be listed in topological block order."
            )
        return cls(layers, discrete_names, continuous_names)

    @property
    def layer_by_name(self):
        return {name: layer for layer, block in enumerate(self.layers)
                for name in block}

    @property
    def discrete_layers(self):
        disc = set(self.discrete_names)
        return tuple(tuple(name for name in block if name in disc)
                     for block in self.layers if block and block[0] in disc)

    @property
    def continuous_block(self):
        continuous = set(self.continuous_names)
        for block in self.layers:
            if block and block[0] in continuous:
                return block
        return tuple()

    def descendants_of(self, action_names):
        if not action_names:
            return set()
        first_action_layer = min(self.layer_by_name[name] for name in action_names)
        return {name for name, layer in self.layer_by_name.items()
                if layer > first_action_layer}
