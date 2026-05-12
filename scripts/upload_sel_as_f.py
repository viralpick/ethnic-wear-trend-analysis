"""local _sel{N}.jpg 를 _f{N}.jpg 이름으로 canonical-thumbnails 에 추가 upload.

SR canonical_object_latest.image_id 의 _f{N} 패턴과 매칭. 기존 _f{N}.jpg (sequential)
는 그대로 두고, _sel{N} 만 _f{N} 이름으로 mirror.
"""
import os, re, time
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

# _sel{N}.jpg 파일만 추출 → _f{N}.jpg 으로 rename upload
sel_files = [f for f in SOURCE_DIR.glob('*.jpg') if re.search(r'_sel\d+\.jpg$', f.name)]
print(f'_sel{{N}}.jpg 파일: {len(sel_files)}')

def upload_one(fp: Path):
    new_name = re.sub(r'_sel(\d+)\.jpg$', r'_f\1.jpg', fp.name)
    try:
        with fp.open('rb') as fh:
            container.upload_blob(
                name=new_name, data=fh, overwrite=True,
                content_settings=ContentSettings(content_type='image/jpeg'),
            )
        return (new_name, True, '')
    except Exception as e:
        return (new_name, False, str(e))

start = time.time()
n_ok = 0; n_fail = 0; failures = []
with ThreadPoolExecutor(max_workers=16) as pool:
    futures = [pool.submit(upload_one, fp) for fp in sel_files]
    for i, fut in enumerate(as_completed(futures), 1):
        name, ok, err = fut.result()
        if ok: n_ok += 1
        else:
            n_fail += 1; failures.append((name, err))
        if i % 1000 == 0:
            print(f'  진행 {i}/{len(sel_files)}')

elapsed = time.time() - start
print(f'\n완료: OK={n_ok}, fail={n_fail} / {elapsed:.1f}s')

# 검증: 처음 BE 가 보낸 URL 확인
import urllib.request
test = 'https://enhansprodpriceagentsa.blob.core.windows.net/canonical-thumbnails/01KQ68Z324RMNB635BG7DDTYHT_01KQ68Z324KE9DTP6H72F8ZKBS_f375.jpg'
try:
    req = urllib.request.Request(test, method='HEAD')
    with urllib.request.urlopen(req, timeout=10) as r:
        print(f'\n검증: BE 의 URL HTTP {r.status} ✓')
        print(f'  {test}')
except Exception as e:
    print(f'\n검증 실패: {e}')
