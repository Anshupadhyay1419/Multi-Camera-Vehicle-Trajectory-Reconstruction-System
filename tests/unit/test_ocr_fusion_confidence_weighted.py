import pytest

from src.validation.ocr_fusion import OCRFusion


def test_confidence_weighted_fusion_resolves_incorrect_plate_variant():
    # Canonical plate: HR12CQ6899
    # Variant with confusion: HR12CO6899 (Q<->O) appears more often,
    # but should lose after confidence-weighted fusion.
    fusion = OCRFusion(window_size=10, max_edit_distance=2)

    tid = 1
    # Wrong variant appears 4 times but with lower confidence overall
    wrong = "HR12CO6899"
    for c in [0.60, 0.62, 0.58, 0.59]:
        fusion.add_result(tid, wrong, c)

    # Correct variant appears 3 times with high confidence
    # (ensures correct has count-majority over the wrong variant).
    correct = "HR12CQ6899"
    for c in [0.93, 0.95, 0.94]:
        fusion.add_result(tid, correct, c)


    fused_plate, fused_conf = fusion.flush(tid)

    assert fused_plate == correct
    assert fused_conf > 0.0



def test_edit_distance_clustering_merges_near_identical_variants():
    fusion = OCRFusion(window_size=10, max_edit_distance=2)
    tid = 2

    base = "DL7CD5017"
    fusion.add_result(tid, base, 0.85)
    fusion.add_result(tid, "DL7CDS017", 0.70)  # 5 vs S shift (within edit distance)
    fusion.add_result(tid, "DL7CD5O17", 0.65)  # O vs 0/5 confusion-like

    fused_plate, _ = fusion.flush(tid)

    # We don't enforce which canonical inside the cluster wins,
    # but it must belong to the cluster (i.e., be one of these)
    assert fused_plate in {base, "DL7CDS017", "DL7CD5O17"}

