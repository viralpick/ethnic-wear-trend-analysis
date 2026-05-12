"""IG image (jpg/heic) → public container `instagram-images` mirror.

source: canonical_object_latest 의 IG image media_ref 의 filename (last segment)
local: sample_data/image_cache/{filename}
dest: container `instagram-images` (public) root, 같은 filename
"""
import os, re, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, 'src')
from loaders.starrocks_connect import connect_result
from azure.storage.blob import BlobServiceClient, ContentSettings, PublicAccess

CONN = os.environ['AZURE_STORAGE_CONNECTION_STRING']
DST_CONTAINER = 'instagram-images'
LOCAL_CACHE = Path('sample_data/image_cache')

svc = BlobServiceClient.from_connection_string(CONN)

# 1. public container 생성/조정
dst = svc.get_container_client(DST_CONTAINER)
if dst.exists():
    acl = dst.get_container_access_policy()
    if acl.get('public_access') != 'blob':
        dst.set_container_access_policy(signed_identifiers={}, public_access=PublicAccess.Blob)
        print('public access → Blob')
    print(f'container {DST_CONTAINER} 이미 존재 (public 확인)')
else:
    svc.create_container(DST_CONTAINER, public_access=PublicAccess.Blob)
    print(f'새 container {DST_CONTAINER} 생성 (public)')

# 2. SR 에서 IG image media_ref 추출 → filename
sr = connect_result()
try:
    with sr.cursor() as cur:
        cur.execute("""
          SELECT DISTINCT co.media_ref
          FROM canonical_object_latest co
          WHERE co.item_source = 'instagram'
            AND co.image_id IS NOT NULL
            AND (co.image_id LIKE '%.heic' OR co.image_id LIKE '%.jpg' OR co.image_id LIKE '%.png')
            AND co.media_ref IS NOT NULL
        """)
        refs = [r['media_ref'] for r in cur.fetchall()]
finally:
    sr.close()

# media_ref 의 last segment = filename
filenames = set()
for url in refs:
    # URL 또는 path 양쪽 호환
    name = url.rsplit('/', 1)[-1].split('?')[0]
    if name:
        filenames.add(name)
print(f'unique filename: {len(filenames)}')

# 3. local cache 매칭 + upload
def upload_one(name: str):
    src = LOCAL_CACHE / name
    if not src.exists():
        return (name, False, 'local missing')
    try:
        with src.open('rb') as fh:
            ct = 'image/heic' if name.endswith('.heic') else (
                 'image/png' if name.endswith('.png') else 'image/jpeg')
            dst.upload_blob(
                name=name, data=fh, overwrite=True,
                content_settings=ContentSettings(content_type=ct),
            )
        return (name, True, '')
    except Exception as e:
        return (name, False, str(e))

start = time.time()
n_ok = 0; n_local_missing = 0; n_fail = 0; failures = []
files = sorted(filenames)
with ThreadPoolExecutor(max_workers=16) as pool:
    futures = [pool.submit(upload_one, n) for n in files]
    for i, fut in enumerate(as_completed(futures), 1):
        name, ok, err = fut.result()
        if ok:
            n_ok += 1
        elif err == 'local missing':
            n_local_missing += 1
        else:
            n_fail += 1
            failures.append((name, err))
        if i % 200 == 0:
            print(f'  진행 {i}/{len(files)}')

elapsed = time.time() - start
print(f'\n완료: OK={n_ok}, local_missing={n_local_missing}, fail={n_fail} / {elapsed:.1f}s')
if failures[:3]:
    for n, e in failures[:3]:
        print(f'  fail {n}: {e[:120]}')

# 4. 검증
import urllib.request
account = re.search(r'AccountName=([^;]+)', CONN).group(1)
sample = next(iter(filenames))
url = f'https://{account}.blob.core.windows.net/{DST_CONTAINER}/{sample}'
try:
    req = urllib.request.Request(url, method='HEAD')
    with urllib.request.urlopen(req, timeout=10) as r:
        print(f'\n검증: anonymous HEAD HTTP {r.status} ✓')
        print(f'  {url}')
except Exception as e:
    print(f'\n검증 실패: {e}')

print(f'\n=== URL 패턴 ===')
print(f'  https://{account}.blob.core.windows.net/{DST_CONTAINER}/{{filename}}')
