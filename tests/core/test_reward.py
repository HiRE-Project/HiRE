"""Numerical checks for the reward primitives used by the experiment shaper."""
import torch
from hire.potential import reference_similarity, contrastive_score
from hire.shaping import potential_difference


def test_similarity_averages_corresponding_patches():
    current = torch.tensor([[[1.0, 0.0], [0.0, 1.0]],
                            [[0.0, 1.0], [1.0, 0.0]]])
    reference = torch.tensor([[[3.0, 0.0], [0.0, 4.0]]])
    # Scaling reference features has no effect on cosine similarity.
    torch.testing.assert_close(reference_similarity(current, reference, 10), torch.tensor([1.0, 0.0]))


def test_missing_failure_references_leave_positive_score_unchanged():
    current = torch.tensor([[[1.0, 0.0]]])
    positive = reference_similarity(current, current, 10)
    negative = reference_similarity(current, None, 10)
    torch.testing.assert_close(contrastive_score(positive, negative, 0.9), positive)


def test_similar_failure_reference_reduces_potential():
    current = torch.tensor([[[1.0, 0.0]]])
    positive = reference_similarity(current, current, 10)
    distant = reference_similarity(current, torch.tensor([[[0.0, 1.0]]]), 10)
    near = reference_similarity(current, current, 10)
    assert (contrastive_score(positive, near) < contrastive_score(positive, distant)).all()


def test_discounted_substep_shaping_matches_chunk_endpoints():
    gamma = 0.99
    phi = torch.tensor([0.2, 0.4, 0.6, 0.9])
    substeps = potential_difference(phi[:-1], phi[1:], gamma)
    chunk = potential_difference(phi[0], phi[-1], gamma ** 3)
    torch.testing.assert_close((substeps * gamma ** torch.arange(3)).sum(), chunk)
