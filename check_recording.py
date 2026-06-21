"""Verify that voice recordings are uploading to Supabase Storage.

Run after the next outbound call ends (~30s after pickup).

Expected output:
  - call_logs table shows recent calls with recording_url
  - Supabase Storage bucket 'outboundai/recordings/' contains matching .ogg files
  - The S3 URLs return HTTP 200 when fetched
"""
import os
import sys
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

import boto3
from supabase import create_client

SUPABASE = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"])
S3 = boto3.client(
    "s3",
    endpoint_url=os.environ["S3_ENDPOINT_URL"],
    aws_access_key_id=os.environ["S3_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["S3_SECRET_ACCESS_KEY"],
    region_name=os.environ.get("S3_REGION", "ap-northeast-1"),
)
BUCKET = os.environ["S3_BUCKET"]


def main() -> int:
    # 1. Latest call_logs rows
    print("=== Latest call_logs (newest first) ===")
    r = (
        SUPABASE.table("call_logs")
        .select("phone_number, outcome, duration_seconds, recording_url, timestamp")
        .order("timestamp", desc=True)
        .limit(5)
        .execute()
    )
    rows = r.data or []
    if not rows:
        print("  (no rows)")
    for c in rows:
        print(f"  {c['timestamp']}  {c['phone_number']:14s}  {c['outcome']:12s}  {c['duration_seconds']}s")
        print(f"    rec: {c.get('recording_url', '(none)')}")

    # 2. Bucket contents
    print()
    print(f"=== Bucket {BUCKET}/recordings/ ===")
    r = S3.list_objects_v2(Bucket=BUCKET, Prefix="recordings/", MaxKeys=50)
    objs = r.get("Contents", [])
    real_files = [o for o in objs if not o["Key"].endswith(".emptyFolderPlaceholder")]
    print(f"  {len(real_files)} real recording(s) + {len(objs) - len(real_files)} placeholder(s)")
    for o in objs:
        size_kb = o["Size"] / 1024
        print(f"  {o['Key']:60s}  {size_kb:8.1f} KB  {o['LastModified']}")

    # 3. Cross-check: does each recent call's predicted URL exist in the bucket?
    print()
    print("=== Cross-check (call recording_url → bucket object) ===")
    import urllib.request, urllib.error
    missing = 0
    for c in rows:
        url = c.get("recording_url") or ""
        if not url:
            continue
        try:
            req = urllib.request.Request(url, method="HEAD")
            with urllib.request.urlopen(req, timeout=10) as resp:
                size = int(resp.headers.get("Content-Length", 0)) // 1024
                ct = resp.headers.get("Content-Type", "?")
                ok = "✓" if resp.status == 200 else "✗"
                print(f"  {ok}  HTTP {resp.status}  {size} KB  {ct}  {url[-60:]}")
        except urllib.error.HTTPError as e:
            missing += 1
            print(f"  ✗  HTTP {e.code}  {url[-60:]}")

    print()
    if missing == 0 and real_files:
        print(f"✅ ALL {len(rows)} recent recordings are reachable")
        return 0
    elif not real_files:
        print("❌ Bucket has no recordings yet — make a test call and re-run")
        return 1
    else:
        print(f"⚠️  {missing} recording URL(s) missing in bucket — egress not uploading")
        return 1


if __name__ == "__main__":
    sys.exit(main())
