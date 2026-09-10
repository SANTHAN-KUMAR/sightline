# Source repository LICENSE: Apache License 2.0, "Copyright 2019 - present, Facebook, Inc" (the repository is a
# Detectron2 fork; the ProbEn functions below were added by the repository authors of "Multimodal Object
# Detection via Probabilistic Ensembling", ECCV 2022). Modifications Copyright 2026 the Sightline authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Vendored from https://github.com/Jamie725/Multimodal-Object-Detection-via-Probabilistic-Ensembling
# (commit 65494f9a16e18fb5c2688bb48ea0adc63ac6ffb6, demo/FLIR/demo_probEn.py: bayesian_fusion,
# bayesian_fusion_multiclass). Modifications (Sightline, 2026-09-10): type hints, probability clipping so a
# score of exactly 0 or 1 cannot produce log(0), a configurable class count (the original hard-coded 3 classes),
# and `proben_single` which states the marginalisation rule explicitly (a box seen by one modality keeps that
# modality's posterior). Score rule only; box fusion stays with ensemble-boxes (WBF).
"""ProbEn probabilistic-ensembling score rule (Chen et al., ECCV 2022)."""

from __future__ import annotations

import numpy as np

__all__ = ["bayesian_fusion", "bayesian_fusion_multiclass", "proben_single"]

_EPS = 1e-7


def bayesian_fusion(match_score_vec) -> float:
    """Fuse the binary (object vs background) scores of one object seen by several detectors.

    Assumes conditional independence of the modalities given the class and a uniform class prior:
    p(y | x1..xn) is proportional to prod_i p(y | xi). Returns the normalised positive posterior.
    """
    s = np.clip(np.asarray(match_score_vec, dtype=np.float64), _EPS, 1 - _EPS)
    fused_positive = np.exp(np.sum(np.log(s)))
    fused_negative = np.exp(np.sum(np.log(1 - s)))
    return float(fused_positive / (fused_positive + fused_negative))


def bayesian_fusion_multiclass(match_score_vec, num_classes: int | None = None) -> tuple[float, int]:
    """Multi-class ProbEn: rows = detectors, columns = per-class scores (background is 1 - sum).

    Returns (fused score of the winning class, winning class index). An index equal to num_classes means
    background won.
    """
    m = np.atleast_2d(np.asarray(match_score_vec, dtype=np.float64))
    k = m.shape[1] if num_classes is None else num_classes
    scores = np.zeros((m.shape[0], k + 1))
    scores[:, :k] = m[:, :k]
    scores[:, -1] = 1 - np.sum(m[:, :k], axis=1)
    scores = np.clip(scores, _EPS, 1.0)
    sum_logits = np.sum(np.log(scores), axis=0)
    score_norm = np.exp(sum_logits - sum_logits.max())
    score_norm /= score_norm.sum()
    return float(np.max(score_norm)), int(np.argmax(score_norm))


def proben_single(score: float) -> float:
    """Marginalisation: a box detected by only one modality keeps that modality's posterior unchanged."""
    return float(score)
