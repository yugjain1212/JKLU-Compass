"""
Index the scraped JKLU website dataset into ChromaDB.

Usage:
    python scripts/index_jklu_dataset.py
    python scripts/index_jklu_dataset.py --batch-size 32
    python scripts/index_jklu_dataset.py --reset
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------
# 1. Configure project paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data"

SOURCE_DIR = DATA_DIR / "raw"
CHROMA_DIR = DATA_DIR / "chroma"

COLLECTION_NAME = "jklu_compass"


# Ensure Python can import the project from src/.
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# Import project components after configuring sys.path.
from jklu_compass.embeddings.encoder import EmbeddingEncoder  # noqa: E402
from jklu_compass.ingestion.indexer import DocumentIndexer  # noqa: E402
from jklu_compass.vectorstore.chroma_store import ChromaVectorStore  # noqa: E402

# ---------------------------------------------------------
# 2. Configure logging
# ---------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# 3. Parse command-line arguments
# ---------------------------------------------------------


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Index the JKLU website dataset into ChromaDB.")

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Number of chunks embedded per batch. Default: 32.",
    )

    operation_group = parser.add_mutually_exclusive_group()

    operation_group.add_argument(
        "--reset",
        action="store_true",
        help="Delete existing records before indexing.",
    )

    operation_group.add_argument(
        "--sync",
        action="store_true",
        help="Remove obsolete chunks from successfully processed sources.",
    )

    return parser.parse_args()


# ---------------------------------------------------------
# 4. Validate the dataset
# ---------------------------------------------------------


def validate_dataset() -> list[Path]:
    """Check the source directory and return Markdown files."""

    if not SOURCE_DIR.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {SOURCE_DIR}")

    if not SOURCE_DIR.is_dir():
        raise NotADirectoryError(f"Expected a directory, received: {SOURCE_DIR}")

    markdown_files = sorted(
        path
        for path in SOURCE_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() in {".md", ".markdown"}
    )

    if not markdown_files:
        raise RuntimeError(f"No Markdown files found in {SOURCE_DIR}")

    logger.info("Project directory: %s", PROJECT_ROOT)
    logger.info("Dataset directory: %s", SOURCE_DIR)
    logger.info("Markdown files found: %d", len(markdown_files))
    logger.info(
        "Dataset size: %.2f MB", sum(path.stat().st_size for path in markdown_files) / (1024 * 1024)
    )

    return markdown_files


# ---------------------------------------------------------
# 5. Initialize the vector store
# ---------------------------------------------------------


def create_vector_store() -> ChromaVectorStore:
    """Create or open the persistent ChromaDB collection."""

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    vector_store = ChromaVectorStore(
        persist_directory=CHROMA_DIR,
        collection_name=COLLECTION_NAME,
    )

    logger.info("ChromaDB directory: %s", CHROMA_DIR)
    logger.info("Collection: %s", COLLECTION_NAME)

    return vector_store


def backup_chroma_directory() -> Path | None:
    """Create a timestamped backup of the existing ChromaDB directory."""

    if not CHROMA_DIR.exists():
        logger.warning("No existing ChromaDB directory found. Backup skipped.")
        return None

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    backup_dir = CHROMA_DIR.with_name(f"{CHROMA_DIR.name}.backup-{timestamp}")

    logger.info("Creating ChromaDB backup...")
    logger.info("Backup destination: %s", backup_dir)

    shutil.copytree(CHROMA_DIR, backup_dir)

    logger.info("ChromaDB backup completed successfully.")

    return backup_dir


# ---------------------------------------------------------
# 6. Index the dataset
# ---------------------------------------------------------


def index_dataset(
    batch_size: int = 32,
    reset: bool = False,
    sync: bool = False,
) -> None:
    """Load, clean, chunk, embed, and index the JKLU dataset."""

    if batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")

    logger.info("=" * 60)
    logger.info("JKLU COMPASS — DATASET INDEXING")
    logger.info("=" * 60)

    markdown_files = validate_dataset()

    # Initialize persistent vector storage.
    if sync:
        backup_chroma_directory()

    # Initialize persistent vector storage.
    vector_store = create_vector_store()

    initial_count = vector_store.count()

    logger.info("Existing vectors: %d", initial_count)

    # Reset only when explicitly requested.
    if reset:
        logger.warning(
            "Reset requested. Removing existing records from the '%s' collection.",
            COLLECTION_NAME,
        )

        vector_store.reset()

        logger.info(
            "Collection reset. Current vector count: %d",
            vector_store.count(),
        )

    elif initial_count > 0 and not sync:
        logger.warning(
            "The collection already contains %d vectors. "
            "Existing records will not be deleted. "
            "Use --sync to remove obsolete chunks from processed sources.",
            initial_count,
        )

    # Load the embedding model.
    logger.info("Loading embedding model...")

    encoder = EmbeddingEncoder()

    logger.info(
        "Embedding dimension: %s",
        encoder.embedding_dimension,
    )
    # Reuse the existing project indexer.
    indexer = DocumentIndexer(
        encoder=encoder,
        vector_store=vector_store,
        batch_size=batch_size,
    )

    logger.info("Starting indexing...")
    logger.info("Batch size: %d", batch_size)

    report = indexer.index_directory(
        SOURCE_DIR,
        sync=sync,
    )

    final_count = vector_store.count()

    # -----------------------------------------------------
    # 7. Print the indexing report
    # -----------------------------------------------------

    logger.info("=" * 60)
    logger.info("INDEXING REPORT")
    logger.info("=" * 60)

    logger.info("Markdown files discovered: %d", len(markdown_files))
    logger.info("Documents loaded:          %d", report.documents_loaded)
    logger.info("Chunks created:             %d", report.chunks_created)
    logger.info("Chunks indexed:             %d", report.chunks_indexed)
    logger.info("Vectors before indexing:    %d", initial_count)
    logger.info("Vectors after indexing:     %d", final_count)
    logger.info("Chunks deleted:             %d", report.chunks_deleted)
    logger.info("Sources synchronized:       %d", report.sources_synced)

    logger.info("Persistent storage: %s", CHROMA_DIR.resolve())
    logger.info("Collection: %s", COLLECTION_NAME)

    # -----------------------------------------------------
    # 8. Validate the result
    # -----------------------------------------------------

    if report.documents_loaded == 0:
        raise RuntimeError("No documents were loaded. Check the dataset and loader.")

    if report.chunks_created == 0:
        raise RuntimeError("No chunks were created. Check the ingestion pipeline.")

    if report.chunks_indexed != report.chunks_created:
        raise RuntimeError(
            f"Indexing incomplete: created {report.chunks_created} "
            f"chunks but indexed {report.chunks_indexed}."
        )

    if final_count == 0:
        raise RuntimeError("ChromaDB contains no vectors after indexing.")

    logger.info("=" * 60)
    logger.info("INDEXING COMPLETED SUCCESSFULLY")
    logger.info("=" * 60)


# ---------------------------------------------------------
# 9. Main entry point
# ---------------------------------------------------------


def main() -> None:
    """Run the indexing process."""

    args = parse_arguments()

    try:
        index_dataset(
            batch_size=args.batch_size,
            reset=args.reset,
            sync=args.sync,
        )

    except KeyboardInterrupt:
        logger.warning("Indexing interrupted by the user.")
        raise SystemExit(130) from None

    except Exception:
        logger.exception("Indexing failed.")
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
