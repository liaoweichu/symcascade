from symcascade.symbolic.skeleton import Skeleton, SkeletonAction
from symcascade.symbolic.matcher import SkeletonMatcher, skeleton_similarity


def _skel(names):
    return Skeleton(actions=tuple(SkeletonAction(name=n) for n in names))


def test_identical_skeletons_have_similarity_one():
    a = _skel(["goto", "take", "goto", "put"])
    b = _skel(["goto", "take", "goto", "put"])
    assert skeleton_similarity(a, b) == 1.0


def test_completely_different_have_zero():
    a = _skel(["goto", "take"])
    b = _skel(["look", "clean"])
    # edit distance 2 over length 2 -> seq_sim 0, no predicate overlap -> sim 0
    assert skeleton_similarity(a, b) == 0.0


def test_partial_overlap_is_between_zero_and_one():
    a = _skel(["goto", "take", "goto", "put"])
    b = _skel(["goto", "take", "goto", "clean"])
    sim = skeleton_similarity(a, b)
    assert 0.0 < sim < 1.0


def test_matcher_returns_best_above_threshold():
    m = SkeletonMatcher(threshold=0.5)
    s1 = _skel(["goto", "take", "goto", "put"])
    s2 = _skel(["goto", "take", "goto", "clean"])
    m.add("s1", s1)
    m.add("s2", s2)
    query = _skel(["goto", "take", "goto", "put"])
    match_id, sim = m.match(query)
    assert match_id == "s1"
    assert sim == 1.0


def test_matcher_returns_none_below_threshold():
    m = SkeletonMatcher(threshold=0.99)
    m.add("s1", _skel(["goto", "take", "goto", "put"]))
    match_id, sim = m.match(_skel(["look", "clean"]))
    assert match_id is None
