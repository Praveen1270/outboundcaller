"""
One-off cleanup script: scan call_logs with recording_url, check each file's
actual WAV duration against call_logs.duration_seconds, and mark any file
that's >1.2x off (corrupted by the old double-track recorder) as bad by
setting recording_url to NULL.
"""
import asyncio
import struct
import time
import httpx
from dotenv import load_dotenv
load_dotenv('.env')
from supabase import create_client
import os

sb = create_client(os.getenv('SUPABASE_URL'), os.getenv('SUPABASE_SERVICE_KEY'))


def wav_duration_seconds(content: bytes):
    """Extract declared duration from a WAV header. Returns None on bad data."""
    try:
        if len(content) < 44:
            return None
        if content[:4] != b'RIFF' or content[8:12] != b'WAVE':
            return None
        # chunk_size at 16-19, audio_format at 20-21, channels at 22-23
        audio_format, num_channels, sample_rate = struct.unpack('<HHI', content[20:28])
        if audio_format != 1:  # only PCM
            return None
        bits = struct.unpack('<H', content[34:36])[0]
        data_idx = content.find(b'data')
        if data_idx < 0 or data_idx + 8 > len(content):
            return None
        data_size = struct.unpack('<I', content[data_idx+4:data_idx+8])[0]
        return data_size / (sample_rate * num_channels * (bits // 8))
    except Exception:
        return None


async def main():
    res = sb.table('call_logs').select('id,phone_number,duration_seconds,recording_url').not_.is_('recording_url', 'null').execute()
    rows = res.data or []
    print(f'Scanning {len(rows)} call_logs with recording_url...')
    bad = []
    checked = 0
    for r in rows[:30]:
        url = r['recording_url']
        if not url:
            continue
        try:
            content = httpx.get(url, timeout=15, follow_redirects=True).content
        except Exception as e:
            print(f'  ⚠ {r["id"][:8]}.. {r["phone_number"]}: download failed — {e}')
            continue
        dur = wav_duration_seconds(content)
        if dur is None:
            continue
        checked += 1
        call_dur = r['duration_seconds'] or 0
        ratio = dur / max(call_dur, 1)
        flag = '⚠ BAD' if ratio > 1.2 or ratio < 0.5 else 'OK  '
        if ratio > 1.2 or ratio < 0.5:
            bad.append((r, dur, call_dur, ratio))
        print(f'  {flag}  {r["id"][:8]}.. {r["phone_number"]:20} call={call_dur:4}s  wav={dur:6.1f}s  ratio={ratio:.2f}')
        time.sleep(0.3)
    print(f'\nChecked {checked} recordings. Found {len(bad)} BAD ones (ratio>1.2 or <0.5).')
    if not bad:
        return
    print('\nMarking bad recordings: setting recording_url=NULL on those rows...')
    for r, wav_d, call_d, ratio in bad:
        try:
            sb.table('call_logs').update({'recording_url': None}).eq('id', r['id']).execute()
            print(f'  ✓ cleared recording_url for {r["id"]} (call={call_d}s, wav={wav_d:.1f}s)')
        except Exception as e:
            print(f'  ✗ failed for {r["id"]}: {e}')


asyncio.run(main())