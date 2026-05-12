"""local _video_thumbs → 새 public container `canonical-thumbnails` 직접 업로드."""
import os, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

from azure.storage.blob import BlobServiceClient, ContentSettings

CONN = os.environ['AZURE_STORAGE_CONNECTION_STRING']
CONTAINER = 'canonical-thumbnails'
SOURCE_DIR = Path('outputs/weekly_review/_video_thumbs')

svc = BlobServiceClient.from_connection_string(CONN)
container = svc.get_container_client(CONTAINER)

files = sorted(SOURCE_DIR.glob('*.jpg'))
print(f'업로드 대상: {len(files)} files')

def upload_one(fp):
    try:
        with fp.open('rb') as fh:
            container.upload_blob(
                name=fp.name, data=fh, overwrite=True,
                content_settings=ContentSettings(content_type='image/jpeg'),
            )
        return (fp.name, True, '')
    except Exception as e:
        return (fp.name, False, str(e))

start = time.time()
n_ok = 0; n_fail = 0; failures = []
with ThreadPoolExecutor(max_workers=16) as pool:
    futures = [pool.submit(upload_one, fp) for fp in files]
    for i, fut in enumerate(as_completed(futures), 1):
        name, ok, err = fut.result()
        if ok: n_ok += 1
        else:
            n_fail += 1; failures.append((name, err))
        if i % 1000 == 0:
            print(f'  진행 {i}/{len(files)} ({i/len(files)*100:.0f}%, {i/(time.time()-start):.0f}/s)')

elapsed = time.time() - start
print(f'\n완료: {n_ok} 성공, {n_fail} 실패 / {elapsed:.1f}s')
if failures:
    for n, e in failures[:3]:
        print(f'  fail {n}: {e[:100]}')

# 검증 — anonymous HEAD
import urllib.request
import re
m = re.search(r'AccountName=([^;]+)', CONN)
account = m.group(1)
sample = files[0].name
url = f'https://{account}.blob.core.windows.net/{CONTAINER}/{sample}'
try:
    req = urllib.request.Request(url, method='HEAD')
    with urllib.request.urlopen(req, timeout=10) as r:
        print(f'\n검증: anonymous HEAD HTTP {r.status} ✓')
        print(f'  {url}')
except Exception as e:
    print(f'\n검증 실패: {e}')

print(f'\n=== 새 URL 패턴 ===')
print(f'  https://{account}.blob.core.windows.net/{CONTAINER}/{{image_id}}.jpg')
