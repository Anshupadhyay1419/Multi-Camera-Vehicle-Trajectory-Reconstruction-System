"""src.validation.ocr_fusion

Multi-frame OCR fusion for the ALPR University Gate system.

This module collects validated plate strings across a sliding window of
frames for each tracked vehicle and fuses them into a single final plate.

Key improvements vs majority voting:
- Confidence-weighted scoring (sum of confidences per cluster) rather than
  count-based majority.
- Similarity-aware merging using edit-distance clustering so common OCR
  confusions (Q↔O, 0↔O, I↔1, etc.) can converge to a single canonical plate.

The caller is responsible for:
- validating plate strings (PlateValidator)
- passing track_id, plate, and confidence
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Iterable, List, Optional, Tuple


def _levenshtein(s1: str, s2: str) -> int:
    """Compute the Levenshtein edit distance between two strings."""
    if s1 == s2:
        return 0
    if len(s1) < len(s2):
        s1, s2 = s2, s1
    if len(s2) == 0:
        return len(s1)

    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            curr.append(
                min(
                    prev[j + 1] + 1,  # deletion
                    curr[j] + 1,  # insertion
                    prev[j] + (0 if c1 == c2 else 1),  # substitution
                )
            )
        prev = curr
    return prev[-1]


@dataclass(frozen=True)
class _Cluster:
    plates: List[str]
    total_conf: float
    best_plate: str


class OCRFusion:
    """Fuse OCR results across multiple frames for each track_id."""

    def __init__(
        self,
        window_size: int = 7,
        *,
        max_edit_distance: int = 2,
        min_confidence: float = 0.0,
        min_exact_votes: int = 1,
    ) -> None:
        """Args:
        window_size: Number of frames to keep per track.
        max_edit_distance: Max edit distance for merging plate variants.
        min_confidence: Optional floor to ignore very low-confidence OCR hits.
        """
        self.window_size = window_size
        self.max_edit_distance = max_edit_distance
        self.min_confidence = min_confidence
        self.min_exact_votes = min_exact_votes

        # track_id -> deque[(plate_string, confidence)]
        self._buffers: Dict[int, Deque[Tuple[str, float]]] = {}

    def add_result(self, track_id: int, plate: str, confidence: float) -> None:
        if track_id not in self._buffers:
            self._buffers[track_id] = deque(maxlen=self.window_size)
        if confidence < self.min_confidence:
            return
        self._buffers[track_id].append((plate, float(confidence)))

    def get_result(self, track_id: int) -> Tuple[str, float] | None:
        buf = self._buffers.get(track_id)
        if not buf:
            return None
        return self._fuse(list(buf))

    def flush(self, track_id: int) -> Tuple[str, float] | None:
        buf = self._buffers.pop(track_id, None)
        if not buf:
            return None
        return self._fuse(list(buf))

    def active_track_ids(self) -> List[int]:
        return [tid for tid, buf in self._buffers.items() if buf]

    def flush_all(self) -> Dict[int, Tuple[str, float]]:
        results: Dict[int, Tuple[str, float]] = {}
        for track_id in list(self._buffers.keys()):
            fused = self.flush(track_id)
            if fused is not None:
                results[track_id] = fused
        return results

    # ------------------------------------------------------------------
    # Internal fusion
    # ------------------------------------------------------------------

    def _fuse(self, entries: List[Tuple[str, float]]) -> Tuple[str, float]:
        """Fuse entries into (plate, fused_confidence).
        
        Uses a hybrid scoring system that balances:
        1. Frequency (how many times a plate appears)
        2. Confidence quality (average confidence per plate)
        
        This prevents high-confidence wrong plates from beating lower-confidence
        correct plates when the wrong plate appears more frequently.
        """
        if not entries:
            return ("", 0.0)
        if len(entries) == 1:
            return entries[0]

        # Build clusters over DISTINCT plate strings.
        # Each distinct plate contributes all its occurrences/confidences.
        plate_to_confs: Dict[str, List[float]] = {}
        for plate, conf in entries:
            plate_to_confs.setdefault(plate, []).append(conf)

        distinct_plates: List[str] = list(plate_to_confs.keys())

        clusters: List[List[str]] = []
        used = [False] * len(distinct_plates)

        # Greedy clustering with seeds ordered by average confidence.
        indexed = list(range(len(distinct_plates)))
        indexed.sort(
            key=lambda i: (sum(plate_to_confs[distinct_plates[i]]) / len(plate_to_confs[distinct_plates[i]])),
            reverse=True,
        )

        for i in indexed:
            if used[i]:
                continue
            seed = distinct_plates[i]
            used[i] = True
            cluster = [seed]

            for j in indexed:
                if used[j]:
                    continue
                cand = distinct_plates[j]
                if abs(len(seed) - len(cand)) > self.max_edit_distance:
                    continue
                if _levenshtein(seed, cand) <= self.max_edit_distance:
                    used[j] = True
                    cluster.append(cand)

            clusters.append(cluster)

        # Score clusters using hybrid metric: frequency * average confidence
        # This prevents wrong plates with high individual confidence from beating
        # correct plates with lower individual confidence but higher frequency.
        # 
        # Example:
        #   Wrong plate: 4 times @ 0.60 avg = score 2.40
        #   Correct plate: 3 times @ 0.94 avg = score 2.82
        # Result: Correct plate wins despite lower frequency.

        best_cluster: Optional[_Cluster] = None
        best_score = -1.0

        for cluster_plates in clusters:
            cluster_confs: List[float] = []
            for p in cluster_plates:
                cluster_confs.extend(plate_to_confs[p])

            total_conf = sum(cluster_confs)
            cluster_count = len(cluster_confs)
            avg_confidence = total_conf / cluster_count if cluster_count > 0 else 0.0

            # Hybrid score: count * average_confidence
            # Heavily weights both frequency and quality
            hybrid_score = cluster_count * avg_confidence

            # Canonical plate: pick plate with highest *total* weighted confidence
            # within the cluster (confidence-weighted majority at cluster level).
            per_plate_sum: Dict[str, float] = {
                p: sum(plate_to_confs[p]) for p in cluster_plates
            }
            per_plate_max: Dict[str, float] = {
                p: max(plate_to_confs[p]) for p in cluster_plates
            }

            # Primary: sum confidence within cluster
            best_plate_sum = max(per_plate_sum.keys(), key=lambda p: per_plate_sum[p])
            # Secondary: max confidence for tie-breaking
            best_plate_max = max(per_plate_max.keys(), key=lambda p: per_plate_max[p])

            # If sums are tied or nearly tied (within 1e-6), select by max conf.
            candidates = list(per_plate_sum.keys())
            max_sum = per_plate_sum[best_plate_sum]
            close = [p for p in candidates if abs(per_plate_sum[p] - max_sum) <= 1e-6]
            best_plate = best_plate_max if len(close) > 1 else best_plate_sum

            # Update best cluster using hybrid score
            if best_cluster is None or hybrid_score > best_score:
                best_cluster = _Cluster(
                    plates=cluster_plates,
                    total_conf=total_conf,
                    best_plate=best_plate,
                )
                best_score = hybrid_score
            elif abs(hybrid_score - best_score) <= 1e-6:
                # Tie-breaker on hybrid score: use higher per-plate sum
                if per_plate_sum[best_plate] > per_plate_sum.get(best_cluster.best_plate, 0.0):
                    best_cluster = _Cluster(
                        plates=cluster_plates,
                        total_conf=total_conf,
                        best_plate=best_plate,
                    )
                    best_score = hybrid_score

        assert best_cluster is not None

        # Similar strings are useful for clustering, but they must not turn a
        # sequence of unrelated OCR guesses into a stored plate.  Requiring
        # exact repeated text is the safe gate for live ALPR deployments.
        if len(plate_to_confs[best_cluster.best_plate]) < self.min_exact_votes:
            return ("", 0.0)

        # Return best cluster's canonical plate and max confidence from all entries
        best_single = max(entries, key=lambda e: e[1])
        return (best_cluster.best_plate, best_single[1])

