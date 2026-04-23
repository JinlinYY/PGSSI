from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Type

import torch
from torch import nn


ACTIVATIONS: dict[str, Type[nn.Module]] = {
    "relu": nn.ReLU,
    "leakyrelu": nn.LeakyReLU,
}


def _build_mlp(
    input_size: int,
    layer_sizes: List[int],
    dropout: float,
    activation: Callable[[], nn.Module],
) -> nn.Sequential:
    modules: List[nn.Module] = []
    in_features = input_size
    for size in layer_sizes:
        modules.append(nn.Linear(in_features, size))
        modules.append(activation())
        if dropout > 0:
            modules.append(nn.Dropout(dropout))
        in_features = size
    return nn.Sequential(*modules)


@dataclass
class NCFConfig:
    num_solutes: int
    num_solvents: int
    embedding_dim: int = 128
    representation_hidden_sizes: List[int] = None
    collaborative_hidden_sizes: List[int] = None
    dropout: float = 0.05
    activation: str = "relu"

    def __post_init__(self) -> None:
        if self.representation_hidden_sizes is None:
            # Two MLP blocks with output sizes of 64 as reported in the paper.
            self.representation_hidden_sizes = [64, 64]
        if self.collaborative_hidden_sizes is None:
            # Three MLP blocks with output sizes 128, 64, and 32.
            self.collaborative_hidden_sizes = [128, 64, 32]
        if self.activation.lower() not in ACTIVATIONS:
            raise ValueError(f"Unsupported activation: {self.activation}")


class RepresentationEncoder(nn.Module):
    """Shared encoder logic for solute and solvent embeddings."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        hidden_sizes: List[int],
        dropout: float,
        activation: Callable[[], nn.Module],
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
        self.mlp = _build_mlp(embedding_dim, hidden_sizes, dropout, activation)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        x = self.embedding(indices)
        x = self.mlp(x)
        return x


class NCFModel(nn.Module):
    """Neural collaborative filtering model for ln γ∞ prediction."""

    def __init__(self, config: NCFConfig) -> None:
        super().__init__()
        activation_cls = ACTIVATIONS[config.activation.lower()]
        self.solute_encoder = RepresentationEncoder(
            num_embeddings=config.num_solutes,
            embedding_dim=config.embedding_dim,
            hidden_sizes=config.representation_hidden_sizes,
            dropout=config.dropout,
            activation=activation_cls,
        )
        self.solvent_encoder = RepresentationEncoder(
            num_embeddings=config.num_solvents,
            embedding_dim=config.embedding_dim,
            hidden_sizes=config.representation_hidden_sizes,
            dropout=config.dropout,
            activation=activation_cls,
        )
        cf_input_size = 2 * config.representation_hidden_sizes[-1] + 1  # include temperature feature
        self.cf_mlp = _build_mlp(cf_input_size, config.collaborative_hidden_sizes, config.dropout, activation_cls)
        self.output_layer = nn.Linear(config.collaborative_hidden_sizes[-1], 1)

    def forward(
        self,
        solute_indices: torch.Tensor,
        solvent_indices: torch.Tensor,
        temperatures: torch.Tensor,
    ) -> torch.Tensor:
        solute_features = self.solute_encoder(solute_indices)
        solvent_features = self.solvent_encoder(solvent_indices)
        combined = torch.cat([solute_features, solvent_features, temperatures], dim=-1)
        latent = self.cf_mlp(combined)
        return self.output_layer(latent)




