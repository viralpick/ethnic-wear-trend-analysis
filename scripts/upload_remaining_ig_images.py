"""HTML 참조 image_cache 1457 file 중 instagram-images container 에 미업로드된 추가분 mirror."""
import os, re, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
from azure.storage.blob import BlobServiceClient, ContentSettings, PublicAccess

CONN = os.environ['AZURE_STORAGE_CONNECTION_STRING']
svc = BlobServiceClient.from_connection_string(CONN)
container = svc.get_container_client('instagram-images')

existing = {b.name for b in container.list_blobs()}
print(f'instagram-images 기존 blob: {len(existing)}')

html = Path('outputs/weekly_review/review_16w.html').read_text(encoding='utf-8')
referenced = set()
for m in re.finditer(r'src="\.\./\.\./sample_data/image_cache/([^"]+)"', html):
    referenced.add(m.group(1))
print(f'HTML 참조 image_cache: {len(referenced)}')

to_upload = referenced - existing
print(f'추가 업로드: {len(to_upload)}')

LOCAL = Path('sample_data/image_cache')

def up(name):
    src = LOCAL / name
    if not src.exists():
        return (name, False, 'local missing')
    try:
        ext = name.rsplit('.', 1)[-1].lower()
        ct = {'heic':'image/heic','png':'image/png','jpg':'image/jpeg','jpeg':'image/jpeg'}.get(ext, 'image/jpeg')
        with src.open('rb') as fh:
            container.upload_blob(name=name, data=fh, overwrite=True,
                content_settings=ContentSettings(content_type=ct))
        return (name, True, '')
    except Exception as e:
        return (name, False, str(e))

start = time.time()
n_ok=0; n_fail=0; failures=[]
with ThreadPoolExecutor(max_workers=16) as pool:
    for name, ok, err in [f.result() for f in [pool.submit(up, n) for n in to_upload]]:
        if ok: n_ok += 1
        else: n_fail += 1; failures.append((name, err))
print(f'완료: OK={n_ok}, fail={n_fail} / {time.time()-start:.1f}s')
for n, e in failures[:5]:
    print(f'  {n}: {e[:100]}')

# 검증
final = {b.name for b in container.list_blobs()}
print(f'\n최종 instagram-images: {len(final)} blobs (HTML 참조 {len(referenced)} 중 cover {len(referenced & final)})')
