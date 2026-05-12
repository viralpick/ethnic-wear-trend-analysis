"""새 public container `canonical-thumbnails` 생성 + 기존 thumbnail server-side copy."""
import os, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
load_dotenv()

from azure.storage.blob import BlobServiceClient, PublicAccess

CONN = os.environ['AZURE_STORAGE_CONNECTION_STRING']
SRC_CONTAINER = 'collectify'
SRC_PREFIX = 'poc/ai_fashion/canonical_thumbnails/'
DST_CONTAINER = 'canonical-thumbnails'

svc = BlobServiceClient.from_connection_string(CONN)

# 1. 새 container 생성 (이미 있으면 skip)
dst = svc.get_container_client(DST_CONTAINER)
if dst.exists():
    print(f'container {DST_CONTAINER} 이미 존재 — public access 만 점검/조정')
    acl = dst.get_container_access_policy()
    if acl.get('public_access') != 'blob':
        dst.set_container_access_policy(signed_identifiers={}, public_access=PublicAccess.Blob)
        print('public access → Blob')
else:
    svc.create_container(DST_CONTAINER, public_access=PublicAccess.Blob)
    print(f'새 container {DST_CONTAINER} 생성 (public_access=Blob)')

# account name (URL 구성용)
import re
m = re.search(r'AccountName=([^;]+)', CONN)
account = m.group(1)
print(f'account: {account}')

# 2. 기존 thumbnail 목록
src = svc.get_container_client(SRC_CONTAINER)
src_blobs = [b.name for b in src.list_blobs(name_starts_with=SRC_PREFIX)]
print(f'복사 대상: {len(src_blobs)} blobs')

def copy_one(name: str) -> tuple[str, bool, str]:
    fname = name[len(SRC_PREFIX):]  # prefix 제거
    src_url = f'https://{account}.blob.core.windows.net/{SRC_CONTAINER}/{name}'
    try:
        dst.get_blob_client(fname).start_copy_from_url(src_url, requires_sync=True)
        return (fname, True, '')
    except Exception as e:
        return (fname, False, str(e))

start = time.time()
n_ok = 0; n_fail = 0; failures = []
with ThreadPoolExecutor(max_workers=24) as pool:
    futures = [pool.submit(copy_one, b) for b in src_blobs]
    for i, fut in enumerate(as_completed(futures), 1):
        name, ok, err = fut.result()
        if ok: n_ok += 1
        else:
            n_fail += 1
            failures.append((name, err))
        if i % 1000 == 0:
            print(f'  진행 {i}/{len(src_blobs)} ({i/len(src_blobs)*100:.0f}%, {i/(time.time()-start):.0f}/s)')

elapsed = time.time() - start
print(f'\n완료: {n_ok} 성공, {n_fail} 실패 / {elapsed:.1f}s')
if failures:
    for n, e in failures[:5]:
        print(f'  fail {n}: {e[:120]}')

# 3. 검증 — 첫 file 의 anonymous URL fetch
import urllib.request
sample = src_blobs[0][len(SRC_PREFIX):]
test_url = f'https://{account}.blob.core.windows.net/{DST_CONTAINER}/{sample}'
try:
    req = urllib.request.Request(test_url, method='HEAD')
    with urllib.request.urlopen(req, timeout=10) as r:
        print(f'\n검증 anonymous HEAD: HTTP {r.status} ✓')
except Exception as e:
    print(f'\n검증 실패: {e}')

print(f'\n=== 새 URL 패턴 ===')
print(f'  https://{account}.blob.core.windows.net/{DST_CONTAINER}/{{image_id}}.jpg')
print(f'  e.g. {test_url}')
