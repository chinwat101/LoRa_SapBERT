#!/usr/bin/env python3
"""
scripts/download_lexitron.py
Download Lexitron 2.0 dataset (telex-utf8.csv) directly from NECTEC Open Data Portal.
URL: https://opend-portal.nectec.or.th/dataset/lexitron-2-0

Usage:
    python scripts/download_lexitron.py
    python scripts/download_lexitron.py --output data/raw/lexitron/telex.csv
"""
import argparse
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Direct download link for Lexitron 2.0 Thai-English (telex-utf8.csv) from NECTEC Open Data Portal
LEXITRON_2_0_TELEX_URL = (
    "https://opend-portal.nectec.or.th/dataset/"
    "bdd85296-9398-499f-b3a7-aab85042d3f9/resource/"
    "6238e10d-3970-47d9-84a4-6b96494ddde7/download/telex-utf8.csv"
)

LEXITRON_2_0_ETLEX_URL = (
    "https://opend-portal.nectec.or.th/dataset/"
    "bdd85296-9398-499f-b3a7-aab85042d3f9/resource/"
    "200da962-2ffe-4cf4-a22d-6e173a5facbe/download/etlex-utf8.csv"
)


def download_file(url: str, dest_path: Path) -> None:
    """Download a file with progress logging."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading from NECTEC Open Data Portal:\n  {url}")
    logger.info(f"Saving to: {dest_path}")

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    req = urllib.request.Request(url, headers=headers)

    with urllib.request.urlopen(req) as response, open(dest_path, "wb") as out_file:
        data = response.read()
        out_file.write(data)
        logger.info(f"✅ Download completed! Size: {len(data):,} bytes ({len(data)/1024/1024:.2f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Lexitron 2.0 dataset from NECTEC Open Data Portal")
    parser.add_argument("--output", default="data/raw/lexitron/telex.csv", help="Path to save telex.csv")
    parser.add_argument("--download-etlex", action="store_true", help="Also download English-Thai etlex-utf8.csv")
    args = parser.parse_args()

    dest = Path(args.output)
    download_file(LEXITRON_2_0_TELEX_URL, dest)

    if args.download_etlex:
        etlex_dest = dest.parent / "etlex.csv"
        download_file(LEXITRON_2_0_ETLEX_URL, etlex_dest)

    logger.info("\nNext step: Run preprocessing:")
    logger.info("  python scripts/preprocess_lexitron.py")


if __name__ == "__main__":
    main()
