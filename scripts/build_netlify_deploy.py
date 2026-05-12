"""Netlify Drop 용 deploy 폴더 빌드.

전략:
- _video_thumbs (722MB / 7146 files) → 이미 Azure Blob 에 업로드됨. HTML 의 src 를
  Azure URL 로 rewrite → zip 에서 제외 (가벼움).
- image_cache (1457 files / 283MB, 참조된 것만) → Netlify zip 의 assets/ 에 복사.
  HTML 의 src 도 assets/ 로 rewrite.

결과:
  outputs/netlify_deploy/
  ├── index.html              (review_16w.html 의 rewritten 사본, default redirect)
  ├── review_16w.html         (rewritten — _video_thumbs → Azure URL, image_cache → assets/)
  └── assets/
      └── image_cache/        (1457 files)
"""
import re, shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEPLOY = ROOT / 'outputs/netlify_deploy'
DEPLOY.mkdir(parents=True, exist_ok=True)

# 1. HTML rewrite
src_html = (ROOT / 'outputs/weekly_review/review_16w.html').read_text(encoding='utf-8')

AZURE_THUMB = 'https://enhansprodpriceagentsa.blob.core.windows.net/canonical-thumbnails'
AZURE_IG = 'https://enhansprodpriceagentsa.blob.core.windows.net/instagram-images'

# _video_thumbs/{name} → Azure canonical-thumbnails container (public)
new_html = re.sub(
    r'src="_video_thumbs/([^"]+)"',
    rf'src="{AZURE_THUMB}/\1"',
    src_html,
)
# ../../sample_data/image_cache/{name} → Azure instagram-images container (public)
# image_cache 까지 Azure 로 빼서 Netlify 에 HTML 22MB 만 호스팅 → bandwidth 부담 1/14
new_html = re.sub(
    r'src="\.\./\.\./sample_data/image_cache/([^"]+)"',
    rf'src="{AZURE_IG}/\1"',
    new_html,
)

# 안내 banner (페이지 상단)
banner = (
    '<div style="background:#e3f2fd;border:1px solid #90caf9;padding:8px 12px;'
    'margin:8px 0;border-radius:6px;font-size:13px;color:#0d47a1;">'
    '🌐 <b>Netlify 호스팅 버전</b> — Azure Blob (video frame) + Netlify (IG image cache) 분리. '
    '검수 / 데모용. 데이터 갱신은 별도 배포 사이클.'
    '</div>'
)
new_html = re.sub(r'(<body[^>]*>)', r'\1' + banner, new_html, count=1)

# review_16w.html + index.html (redirect) 둘 다 저장
(DEPLOY / 'review_16w.html').write_text(new_html, encoding='utf-8')
index = (
    '<!doctype html><html><head><meta charset="utf-8">'
    '<meta http-equiv="refresh" content="0; url=review_16w.html"></head>'
    '<body>Redirecting to <a href="review_16w.html">review_16w.html</a>…</body></html>'
)
(DEPLOY / 'index.html').write_text(index, encoding='utf-8')
print(f'HTML rewrite OK → {DEPLOY}/review_16w.html ({len(new_html)/1024/1024:.1f} MB)')

# image_cache 도 Azure 로 빠짐 → Netlify deploy 폴더에 별도 copy 불필요
# (이전 assets/image_cache/ 폴더 cleanup)
import shutil as _sh
_old_assets = DEPLOY / 'assets'
if _old_assets.exists():
    _sh.rmtree(_old_assets)
    print(f'옛 assets/ 폴더 삭제 (image_cache 도 Azure 사용)')

# 3. 사이즈 정리
import subprocess
result = subprocess.run(['du', '-sh', str(DEPLOY)], capture_output=True, text=True)
print(f'\n=== {DEPLOY} 총 사이즈 ===')
print(result.stdout.strip())
print()
print('Netlify Drop 안내:')
print(f'  1. https://app.netlify.com/drop 접속')
print(f'  2. {DEPLOY} 폴더를 drop')
print(f'  3. 발급되는 *.netlify.app URL 받음')
