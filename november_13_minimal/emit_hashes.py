"""Self-audit: recompute every VERIFY_HASH line from the saved metrics exactly the way the
instructor audit guide describes (fnv1a over task_id|student_id|P_log|metric1|metric2)."""
import json
import multiprocessing as mp
import sys
from pathlib import Path

from harness import emit_verify_tag, fnv1a_hash
from labutil import resolve_student_id

PHASES = [("PHASE0", "phase0.json"), ("PHASE1", "phase1.json"), ("PHASE2", "phase2.json"),
          ("PHASE3", "phase3.json"), ("PHASE4", "phase4.json")]


def main() -> None:
    results = Path(__file__).resolve().parent / (sys.argv[1] if len(sys.argv) > 1 else "results")
    sid = resolve_student_id(None)
    for task_id, fname in PHASES:
        data = json.loads((results / fname).read_text())
        primary, secondary = data["hash_inputs"]["primary"], data["hash_inputs"]["secondary"]
        payload = f"{task_id}|{sid}|{mp.cpu_count()}|{primary:.4f}|{secondary:.4f}"
        recomputed = emit_verify_tag(task_id, sid, primary, secondary)
        status = "OK " if recomputed == data["verify"] else "MISMATCH"
        print(f"{status} {recomputed}   payload={payload!r}  fnv1a={fnv1a_hash(payload):016X}")


if __name__ == "__main__":
    main()
