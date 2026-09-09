"""The core package must import without any optional extra and must not pull torch/langchain."""

from __future__ import annotations

import subprocess
import sys


def test_core_import_is_light() -> None:
    code = (
        "import sys\n"
        "import tombstone, tombstone.cli, tombstone.config, tombstone.lineage.store, "
        "tombstone.receipt.ledger, tombstone.erase.journal, tombstone.model\n"
        "import tombstone.commands\n"
        "heavy = [m for m in sys.modules if m.split('.')[0] in "
        "{'torch','langchain','langchain_core','transformers','peft','chromadb','faiss',"
        "'qdrant_client','psycopg','numpy','sentence_transformers','mcp','matplotlib'}]\n"
        "print(','.join(sorted(set(m.split('.')[0] for m in heavy))))\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert proc.stdout.strip() == "", f"core import pulled in heavy modules: {proc.stdout.strip()}"
