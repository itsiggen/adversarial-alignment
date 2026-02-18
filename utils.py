import subprocess
import sys
from typing import List, Tuple


def get_free_gpus(min_free_gb: float = 16.0) -> List[Tuple[int, float]]:
    """Return (gpu_id, free_gb) for GPUs with enough free memory, sorted most-free first."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True,
        )
        gpus = []
        for line in result.stdout.strip().split("\n"):
            if not line.strip():
                continue
            parts = line.split(",")
            gpu_id = int(parts[0])
            free_gb = (float(parts[2]) - float(parts[1])) / 1024.0
            if free_gb >= min_free_gb:
                gpus.append((gpu_id, free_gb))
        gpus.sort(key=lambda x: x[1], reverse=True)
        return gpus
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError) as e:
        print(f"Error querying GPUs: {e}", file=sys.stderr)
        return []
