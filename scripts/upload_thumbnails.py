"""canonical_thumbnails 7146 files → Azure Blob `collectify/poc/ai_fashion/canonical_thumbnails/`."""
import os, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

from azure.storage.blob import BlobServiceClient, ContentSettings

CONN = os.environ['AZURE_STORAGE_CONNECTION_STRING']
CONTAINER = os.environ.get('AZURE_STORAGE_CONTAINER', 'collectify')
PREFIX = 'poc/ai_fashion/canonical_thumbnails'
SOURCE_DIR = Path('outputs/weekly_review/_video_thumbs')

service = BlobServiceClient.from_connection_string(CONN)
container = service.get_container_client(CONTAINER)

try:
    container.get_container_properties()
except Exception as e:
    print(f'WARN container access: {e}')

files = sorted(SOURCE_DIR.glob('*.jpg'))
print(f'업로드 대상: {len(files)} files')

def upload_one(fp):
    blob_name = f'{PREFIX}/{fp.name}'
    try:
        with fp.open('rb') as fh:
            container.upload_blob(
                name=blob_name, data=fh, overwrite=True,
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
        if ok:
            n_ok += 1
        else:
            n_fail += 1
            failures.append((name, err))
        if i % 500 == 0:
            elapsed = time.time() - start
            rate = i / elapsed
            print(f'  진행 {i}/{len(files)} ({i/len(files)*100:.0f}%, {rate:.0f} file/s)')

elapsed = time.time() - start
print()
print(f'완료: {n_ok} 성공, {n_fail} 실패 / {elapsed:.1f}s ({n_ok/elapsed:.0f} file/s)')
if failures:
    print('실패 sample:')
    for n, e in failures[:5]:
        print(f'  {n}: {e[:80]}')

print()
print('=== URL 패턴 (BE 한테 공유) ===')
print(f'  https://{{ACCOUNT_NAME}}.blob.core.windows.net/{CONTAINER}/{PREFIX}/{{image_id}}.jpg')
print(f'  e.g. {files[0].name} → /{CONTAINER}/{PREFIX}/{files[0].name}')
