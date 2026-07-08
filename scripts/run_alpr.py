"""ALPR - processes video and saves one entry per vehicle to CSV."""
from __future__ import annotations
import argparse, csv, re, sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import cv2
sys.path.insert(0, str(Path(__file__).parent.parent))

def _lev(s1, s2):
    if s1 == s2: return 0
    if len(s1) < len(s2): s1, s2 = s2, s1
    if not s2: return len(s1)
    prev = list(range(len(s2)+1))
    for c1 in s1:
        curr = [prev[0]+1]
        for j, c2 in enumerate(s2):
            curr.append(min(prev[j+1]+1, curr[j]+1, prev[j]+(c1!=c2)))
        prev = curr
    return prev[-1]

def _fix_ocr(raw, validator):
    """Validate raw OCR string and return canonical plate (or None).
    
    No O->Q correction here — that is handled by _resolve_canonical()
    using multi-frame evidence across the full vehicle pass.
    """
    plate, _ = validator.validate(raw)
    return plate

def _character_vote(plates_with_conf, validator):
    """
    Given a list of (plate_string, confidence) from multiple frames,
    determine the best plate using per-character majority voting.
    
    This handles O/Q confusion correctly:
    - If 40 frames read 'O' and 20 frames read 'Q' at position 5,
      majority = 'O', so keep 'O' (it's a real O plate)
    - If 10 frames read 'O' and 25 frames read 'Q' at position 5,
      majority = 'Q', so use 'Q' (OCR was reading Q wrong as O)
    
    This is safe: a plate with real 'O' will consistently read 'O' across
    most frames. A plate with 'Q' will sometimes read 'O' and sometimes 'Q'.
    """
    if not plates_with_conf:
        return None
    
    # Only apply per-character voting for 10-char plates (standard format)
    # For non-standard lengths, just take the most-read plate
    lengths = set(len(p) for p, _ in plates_with_conf)
    if len(lengths) != 1 or list(lengths)[0] != 10:
        # Mixed lengths or non-standard: return most frequent
        from collections import Counter
        counts = Counter(p for p, _ in plates_with_conf)
        return counts.most_common(1)[0][0]
    
    # Per-position character voting weighted by confidence
    # For each position, count how many times each character appears
    from collections import defaultdict
    pos_char_conf = [defaultdict(float) for _ in range(10)]
    pos_char_count = [defaultdict(int) for _ in range(10)]
    
    for plate, conf in plates_with_conf:
        for i, ch in enumerate(plate):
            pos_char_conf[i][ch] += conf
            pos_char_count[i][ch] += 1
    
    # For each position, pick the character with highest total confidence
    best_chars = []
    for i in range(10):
        if not pos_char_conf[i]:
            return None
        best_ch = max(pos_char_conf[i], key=lambda c: pos_char_conf[i][c])
        best_chars.append(best_ch)
    
    voted = "".join(best_chars)
    plate, _ = validator.validate(voted)
    if plate:
        return plate
    
    # Fallback: return the most frequent plate
    from collections import Counter
    counts = Counter(p for p, _ in plates_with_conf)
    return counts.most_common(1)[0][0]


def _resolve(reads, min_reads, max_dist):
    if not reads: return None
    plates = sorted(reads, key=lambda p: len(reads[p]), reverse=True)
    assigned, merged = set(), {}
    for plate in plates:
        if plate in assigned: continue
        group = list(reads[plate]); assigned.add(plate)
        for other in plates:
            if other in assigned: continue
            if abs(len(plate)-len(other)) > max_dist: continue
            if _lev(plate, other) <= max_dist:
                group.extend(reads[other]); assigned.add(other)
        merged[plate] = group
    best, best_score = None, -1.0
    for plate, entries in merged.items():
        if len(entries) < min_reads: continue
        confs = [e[0] for e in entries]
        score = len(entries) * (sum(confs)/len(confs))
        if score > best_score: best_score, best = score, plate
    if best is None: return None
    entries = merged[best]
    best_conf = max(e[0] for e in entries)
    best_total = len(entries)
    best_frame = min(e[1] for e in entries)
    return (best, best_conf, best_total, best_frame)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", default="output/results.csv")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--min-reads", type=int, default=2)
    parser.add_argument("--max-edit-distance", type=int, default=2)
    parser.add_argument("--max-gap-frames", type=int, default=13)
    args = parser.parse_args()

    from src.utils.config import load_config
    from src.utils.logger import get_logger
    from src.validation.plate_validator import PlateValidator
    from ultralytics import YOLO
    from paddleocr import PaddleOCR

    config = load_config(args.config)
    log = get_logger("run_alpr", config=config)
    validator = PlateValidator()
    plate_model = YOLO(config["detection"]["plate_model_path"])
    ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
    log.info("Models loaded. Video: %s", args.source)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        log.error("Cannot open: %s", args.source); sys.exit(1)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    log.info("Total frames: %d", total)

    frame_hits = defaultdict(list)
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_idx += 1
        h_f, w_f = frame.shape[:2]
        results = plate_model(frame, verbose=False, conf=0.20)
        for r in results:
            if r.boxes is None or len(r.boxes) == 0: continue
            for box in r.boxes:
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w_f, x2), min(h_f, y2)
                if x2 <= x1 or y2 <= y1: continue
                pcrop = frame[y1:y2, x1:x2].copy()
                if pcrop.size == 0: continue
                ph, pw = pcrop.shape[:2]
                scale = max(200.0/pw if pw < 200 else 1.0, 64.0/ph if ph < 64 else 1.0)
                if scale > 1.0:
                    pcrop = cv2.resize(pcrop, (int(pw*scale), int(ph*scale)), interpolation=cv2.INTER_CUBIC)
                result = ocr.ocr(pcrop, cls=True)
                if not result or not result[0]: continue
                texts, confs = [], []
                for line in result[0]:
                    if line:
                        t, c = line[1]
                        texts.append(t)
                        confs.append(float(c))
                if not texts: continue
                raw = re.sub(r"[^A-Z0-9]", "", "".join(texts).upper().replace(" ", ""))
                raw = raw.replace("IND", "").replace("INDIA", "")
                if not raw: continue
                avg_conf = sum(confs) / len(confs)
                if avg_conf < 0.25: continue
                plate = _fix_ocr(raw, validator)
                if plate is None: continue
                frame_hits[frame_idx].append((plate, avg_conf))
                log.debug("Frame %04d  %s  %.2f", frame_idx, plate, avg_conf)

    cap.release()
    log.info("Scan done. Frames with hits: %d / %d", len(frame_hits), frame_idx)

    hit_frames = sorted(frame_hits)
    if not hit_frames:
        print("No plates detected.")
        return

    passes, current = [], [hit_frames[0]]
    for f in hit_frames[1:]:
        if f - current[-1] <= args.max_gap_frames:
            current.append(f)
        else:
            passes.append(current)
            current = [f]
    passes.append(current)
    log.info("Vehicle passes: %d", len(passes))

    results = []
    for pf in passes:
        pr = defaultdict(list)
        for f in pf:
            for plate, conf in frame_hits[f]:
                pr[plate].append((conf, f))
        out = _resolve(pr, args.min_reads, args.max_edit_distance)
        if out is None:
            log.info("Pass %d-%d skipped", pf[0], pf[-1])
            continue
        best_group_plate, conf, reads, first = out

        # Character-level voting within the merged group to handle O/Q ambiguity
        # This uses multi-frame evidence: if majority of frames read Q, use Q
        # If majority read O, keep O (it might genuinely be O)
        group_entries = []
        for orig_plate in pr:
            if _lev(orig_plate, best_group_plate) <= args.max_edit_distance:
                for c, f in pr[orig_plate]:
                    group_entries.append((orig_plate, c))
        voted_plate = _character_vote(group_entries, validator)
        plate = voted_plate if voted_plate else best_group_plate
        _, series = validator.validate(plate)
        log.info("VEHICLE: %s | reads=%d | conf=%.2f | frames %d-%d",
                 plate, reads, conf, pf[0], pf[-1])
        results.append({
            "plate_number": plate,
            "vehicle_type": "Private",
            "plate_color": "White",
            "series_type": series or "normal",
            "confidence": f"{conf:.2f}",
            "total_reads": reads,
            "best_frame": first,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

    results.sort(key=lambda r: r["best_frame"])
    if results:
        fields = ["plate_number", "vehicle_type", "plate_color", "series_type",
                  "confidence", "total_reads", "best_frame", "timestamp"]
        with open(args.output, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(results)
        log.info("Saved %d row(s) to %s", len(results), args.output)

    print(f"\n{'='*50}\n  RESULTS: {len(results)} vehicle(s)\n{'='*50}")
    for r in results:
        print(f"  {r['plate_number']:15s}  reads={r['total_reads']:>3}  conf={r['confidence']}")
    print(f"\n  Saved to: {args.output}\n")

if __name__ == "__main__":
    main()
