#!/usr/bin/env python3
"""
Benchmark OCR accuracy on a folder of test images.
Usage: python scripts/benchmark.py --folder ./test_images --api-key dl_live_xxx
"""
import asyncio
import argparse
import httpx
import json
import os
import time
from pathlib import Path


async def benchmark(folder: str, api_url: str, api_key: str):
    """Run benchmark on test images."""
    folder = Path(folder)
    if not folder.exists():
        print(f"Folder not found: {folder}")
        return

    # Find images
    extensions = {".jpg", ".jpeg", ".png", ".tiff", ".pdf"}
    files = [f for f in folder.iterdir() if f.suffix.lower() in extensions]

    if not files:
        print(f"No image files found in {folder}")
        return

    print(f"Found {len(files)} files to process")
    print("=" * 70)

    results = []
    async with httpx.AsyncClient(timeout=60.0) as client:
        for i, filepath in enumerate(files, 1):
            print(f"[{i}/{len(files)}] Processing: {filepath.name}...", end=" ", flush=True)

            start = time.time()
            try:
                with open(filepath, "rb") as f:
                    response = await client.post(
                        f"{api_url}/api/v1/recognize",
                        headers={"X-API-Key": api_key},
                        files={"file": (filepath.name, f, "image/jpeg")},
                    )

                elapsed = (time.time() - start) * 1000
                data = response.json()

                if response.status_code == 200:
                    doc_type = data.get("document_type", "unknown")
                    confidence = data.get("overall_confidence", 0)
                    fields_count = len(data.get("fields", {}))
                    warnings = len(data.get("warnings", []))

                    print(f"OK | {doc_type} | conf={confidence:.2f} | {fields_count} fields | {elapsed:.0f}ms")
                    results.append({
                        "file": filepath.name,
                        "status": "ok",
                        "document_type": doc_type,
                        "confidence": confidence,
                        "fields_count": fields_count,
                        "warnings": warnings,
                        "time_ms": elapsed,
                    })
                else:
                    print(f"ERROR {response.status_code}: {data}")
                    results.append({
                        "file": filepath.name,
                        "status": "error",
                        "error": str(data),
                        "time_ms": elapsed,
                    })

            except Exception as e:
                elapsed = (time.time() - start) * 1000
                print(f"FAIL: {e}")
                results.append({
                    "file": filepath.name,
                    "status": "fail",
                    "error": str(e),
                    "time_ms": elapsed,
                })

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    ok_results = [r for r in results if r["status"] == "ok"]
    if ok_results:
        avg_conf = sum(r["confidence"] for r in ok_results) / len(ok_results)
        avg_time = sum(r["time_ms"] for r in ok_results) / len(ok_results)
        avg_fields = sum(r["fields_count"] for r in ok_results) / len(ok_results)

        print(f"Total files:      {len(results)}")
        print(f"Successful:       {len(ok_results)}")
        print(f"Failed:           {len(results) - len(ok_results)}")
        print(f"Avg confidence:   {avg_conf:.2f}")
        print(f"Avg fields:       {avg_fields:.1f}")
        print(f"Avg time:         {avg_time:.0f}ms")

        # By document type
        types = {}
        for r in ok_results:
            t = r["document_type"]
            if t not in types:
                types[t] = []
            types[t].append(r)

        print(f"\nBy document type:")
        for t, rs in sorted(types.items()):
            avg_c = sum(r["confidence"] for r in rs) / len(rs)
            print(f"  {t}: {len(rs)} docs, avg confidence {avg_c:.2f}")

    # Save results
    output = folder / "benchmark_results.json"
    with open(output, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nDetailed results saved to: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark DocLens OCR")
    parser.add_argument("--folder", required=True, help="Folder with test images")
    parser.add_argument("--api-url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--api-key", required=True, help="API key")
    args = parser.parse_args()

    asyncio.run(benchmark(args.folder, args.api_url, args.api_key))
