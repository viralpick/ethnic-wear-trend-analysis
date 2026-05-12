"""누락된 video frame thumbnail 추출 + Azure Blob 업로드.

Plan:
1. SR canonical_object_latest 에서 image_id IS NOT NULL 인 video frame 중,
   canonical-thumbnails container 에 없는 image_id 들 추출.
2. media_ref (video URL) → Azure Blob path 변환 → connection string 으로 직접 fetch
   (SAS expire 무관).
3. 각 video 별 cv2.VideoCapture → 필요한 frame_index 들만 seek + extract.
4. 추출된 frame 을 `outputs/weekly_review/_video_thumbs/{image_id}.jpg` 로 disk save
   + canonical-thumbnails container 에 직접 upload.
"""
import os, re, sys, time, tempfile
from collections import defaultdict
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, 'src')
from loaders.starrocks_connect import connect_result
from azure.storage.blob import BlobServiceClient, ContentSettings
import cv2

CONN = os.environ['AZURE_STORAGE_CONNECTION_STRING']
DST_CONTAINER = 'canonical-thumbnails'
LOCAL_DIR = Path('outputs/weekly_review/_video_thumbs')

svc = BlobServiceClient.from_connection_string(CONN)
dst = svc.get_container_client(DST_CONTAINER)

# 1. SR 에서 video frame image_id + media_ref
print('[1/4] SR query — video frame image_id 추출')
sr = connect_result()
try:
    with sr.cursor() as cur:
        cur.execute("""
          SELECT image_id, media_ref, item_source, frame_index
          FROM canonical_object_latest
          WHERE image_id LIKE '%_f%'
            AND image_id NOT LIKE '%.jpg' AND image_id NOT LIKE '%.heic' AND image_id NOT LIKE '%.png'
            AND media_ref IS NOT NULL
            AND frame_index IS NOT NULL
        """)
        all_rows = cur.fetchall()
finally:
    sr.close()
print(f'  total: {len(all_rows)}')

# 2. 컨테이너에 이미 있는 file 들 list (한 번에)
print('[2/4] 컨테이너 의 기존 thumbnail 목록')
existing = {b.name.removesuffix('.jpg') for b in dst.list_blobs() if b.name.endswith('.jpg')}
print(f'  기존 blob: {len(existing)}')

# 3. 누락 image_id 별 grouping (같은 video 의 여러 frame 묶기)
missing_by_video: dict[str, list] = defaultdict(list)  # blob_name → [(image_id, frame_index), ...]
for r in all_rows:
    img_id = r['image_id']
    if img_id in existing:
        continue
    media = r['media_ref']
    # media_ref 형식: "collectify/poc/ai_fashion/instagram/videos/...mp4" 또는 full URL
    # path 추출 (container/blob)
    m = re.match(r'(?:https?://[^/]+/)?(?P<container>[^/]+)/(?P<blob>.+?)(?:\?.*)?$', media)
    if not m:
        continue
    container = m.group('container')
    blob_path = m.group('blob')
    missing_by_video[(container, blob_path)].append((img_id, int(r['frame_index'])))

n_video = len(missing_by_video)
n_frame = sum(len(v) for v in missing_by_video.values())
print(f'  누락 video: {n_video}, 누락 frame: {n_frame}')

# 4. 각 video 의 frame extract + upload
print('[3/4] video 다운로드 → frame extract → upload')
n_video_ok = 0; n_video_fail = 0
n_frame_ok = 0; n_frame_fail = 0
fail_videos = []

def process_video(item):
    (container, blob_path), frames = item
    local_n_ok = 0; local_n_fail = 0
    err_msg = ''
    try:
        # video 다운로드 (connection string auth, SAS 무관)
        bc = svc.get_blob_client(container=container, blob=blob_path)
        with tempfile.NamedTemporaryFile(suffix=Path(blob_path).suffix, delete=False) as tmp:
            data = bc.download_blob().readall()
            tmp.write(data)
            tmp_path = tmp.name
        try:
            cap = cv2.VideoCapture(tmp_path)
            if not cap.isOpened():
                return (container, blob_path, 0, len(frames), 'cv2 open fail')
            for img_id, frame_idx in frames:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ok, bgr = cap.read()
                if not ok or bgr is None:
                    local_n_fail += 1
                    continue
                # disk save
                local_path = LOCAL_DIR / f'{img_id}.jpg'
                cv2.imwrite(str(local_path), bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
                # blob upload
                with local_path.open('rb') as fh:
                    dst.upload_blob(
                        name=f'{img_id}.jpg', data=fh, overwrite=True,
                        content_settings=ContentSettings(content_type='image/jpeg'),
                    )
                local_n_ok += 1
            cap.release()
        finally:
            try: os.unlink(tmp_path)
            except OSError: pass
    except Exception as e:
        return (container, blob_path, local_n_ok, len(frames) - local_n_ok, str(e)[:200])
    return (container, blob_path, local_n_ok, local_n_fail, err_msg)

start = time.time()
with ThreadPoolExecutor(max_workers=6) as pool:
    futures = [pool.submit(process_video, item) for item in missing_by_video.items()]
    for i, fut in enumerate(as_completed(futures), 1):
        container, blob_path, ok, fail, err = fut.result()
        n_frame_ok += ok
        n_frame_fail += fail
        if ok > 0:
            n_video_ok += 1
        if fail > 0 or err:
            if err:
                fail_videos.append((blob_path, err))
        if i % 20 == 0:
            elapsed = time.time() - start
            print(f'  진행 {i}/{n_video} ({i/n_video*100:.0f}%, {i/elapsed:.1f} video/s, frames OK={n_frame_ok})')

elapsed = time.time() - start
print()
print(f'[4/4] 결과 / {elapsed:.1f}s')
print(f'  video 처리: OK={n_video_ok}, fail+={len(fail_videos)}')
print(f'  frame 추출: OK={n_frame_ok}, fail={n_frame_fail}')
if fail_videos[:5]:
    print(f'  fail sample:')
    for bp, e in fail_videos[:5]:
        print(f'    {bp[-60:]}: {e[:120]}')
