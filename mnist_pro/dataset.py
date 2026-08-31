"""Episode sampling. Deterministic given a seed, and independent of torchvision
at analysis time: an episode records the source indices, so a canvas can always be
rebuilt from them."""

from __future__ import annotations

import collections
import random
from dataclasses import dataclass


@dataclass(frozen=True)
class EpisodeSpec:
    episode_id: int
    indices: tuple
    labels: tuple

    @property
    def label(self) -> str:
        return "".join(str(x) for x in self.labels)


def load_mnist(data_dir: str = "data", train: bool = False):
    from torchvision import datasets

    return datasets.MNIST(root=data_dir, train=train, download=True)


def sample_balanced(dataset, num_sets: int, digits: int = 1, seed: int = 42):
    """`num_sets` balanced sets. One digit: each set is one image per class 0-9.
    More than one: each set is ten sequences of uniformly drawn labels.

    Sampling reproduces the original drivers exactly, so episode ids continue to
    refer to the same images as the released runs.
    """
    old = random.getstate()
    random.seed(seed)
    try:
        by_class = collections.defaultdict(list)
        for idx, (_, label) in enumerate(dataset):
            by_class[label].append(idx)

        specs: list[EpisodeSpec] = []
        if digits == 1:
            for _ in range(num_sets):
                for label in range(10):
                    idx = random.choice(by_class[label])
                    specs.append(EpisodeSpec(len(specs), (idx,), (label,)))
        else:
            for _ in range(num_sets * 10):
                labels = tuple(random.randint(0, 9) for _ in range(digits))
                indices = tuple(random.choice(by_class[l]) for l in labels)
                specs.append(EpisodeSpec(len(specs), indices, labels))
        return specs
    finally:
        random.setstate(old)


def images_for(dataset, spec: EpisodeSpec):
    return [dataset[i][0] for i in spec.indices]
