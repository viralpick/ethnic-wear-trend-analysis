"""Color space 유틸 — RGB↔LAB 변환, skin 제거, KMeans 팔레트 추출.

출처: `~/dev/clothing-color-extraction-poc/scripts/color_utils.py` (2026-04-17 동료 PoC).
100% 인수인계 받은 상태이므로 우리 레포에 cherry-copy 후 독자 유지보수.

설계 원칙:
- pure 함수, 외부 I/O 없음
- numpy + scikit-learn 외 의존성 없음 — `[project.optional-dependencies].color` 로 격리
- SKIN_LAB_MIN/MAX 는 모듈 기본값. 방법론 실험 시 함수 인자로 override 가능
  (config 주입 경로: configs/local.yaml `vision.skin_lab_box` → VisionConfig → 호출부)
- KMeans 는 random_state=0 고정 — 같은 입력 → 같은 출력 (snapshot 테스트 대응)

참고 (spec §4.1 ④, §7):
- 동료 PoC verdict: VLM 단독 실패 (ΔE mean 20~25), Pipeline B (segformer + LAB KMeans) 채택
- drop_skin 은 segmentation 의 "belt-and-suspenders". segformer 를 쓰지 않는 방법론
  (bbox-only, rembg 등) 에서는 drop_skin 이 주 방어선이 된다 — 제거율이 방법론 따라 변함
- hex_skin_leak 은 dominant 가 skin box 안에 드는지 최종 QA 플래그
"""
from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans

# --------------------------------------------------------------------------- #
# Skin LAB box 기본값 (spec §4.1 ④ Plan B — Pipeline B 기본)
# --------------------------------------------------------------------------- #
# L (40~80): 어두운~밝은 피부. deep skin (L<40) 은 박스 밖이라 leak 가능 — tune 필요시 낮추기
# a (10~25): 붉은 톤 (피부의 red cast)
# b (15~35): 노란 톤 (피부의 yellow cast)
SKIN_LAB_MIN: np.ndarray = np.array([40.0, 10.0, 15.0])
SKIN_LAB_MAX: np.ndarray = np.array([80.0, 25.0, 35.0])


# --------------------------------------------------------------------------- #
# 기본 변환 (HEX / RGB / LAB)
# --------------------------------------------------------------------------- #

def rgb_to_hex(rgb: np.ndarray) -> str:
    """RGB (0~255) → '#RRGGBB' 대문자. 범위 밖은 clip."""
    rgb = np.clip(rgb, 0, 255).astype(int)
    return "#{:02X}{:02X}{:02X}".format(*rgb.tolist())


def hex_to_rgb(hx: str) -> np.ndarray:
    """'#RRGGBB' 또는 'RRGGBB' → np.array([r, g, b]) float32."""
    hx = hx.lstrip("#")
    return np.array([int(hx[i:i + 2], 16) for i in (0, 2, 4)], dtype=np.float32)


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """RGB (0~255, uint8 or float) → LAB (D65 / sRGB gamma). (...,3) 유지."""
    rgb = np.asarray(rgb, dtype=np.float32) / 255.0
    mask = rgb > 0.04045
    lin = np.where(mask, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)
    M = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ], dtype=np.float32)
    xyz = lin @ M.T
    white = np.array([0.95047, 1.0, 1.08883], dtype=np.float32)
    xyz = xyz / white
    eps = 216 / 24389
    kappa = 24389 / 27
    f = np.where(xyz > eps, np.cbrt(xyz), (kappa * xyz + 16) / 116)
    L = 116 * f[..., 1] - 16
    a = 500 * (f[..., 0] - f[..., 1])
    b = 200 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], axis=-1)


def lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    """LAB → RGB 0~255 (벡터화). 범위 밖은 clip."""
    lab = np.asarray(lab, dtype=np.float32)
    fy = (lab[..., 0] + 16) / 116
    fx = lab[..., 1] / 500 + fy
    fz = fy - lab[..., 2] / 200
    eps = 216 / 24389
    kappa = 24389 / 27
    xyz = np.stack([
        np.where(fx ** 3 > eps, fx ** 3, (116 * fx - 16) / kappa),
        np.where(lab[..., 0] > kappa * eps, ((lab[..., 0] + 16) / 116) ** 3, lab[..., 0] / kappa),
        np.where(fz ** 3 > eps, fz ** 3, (116 * fz - 16) / kappa),
    ], axis=-1)
    white = np.array([0.95047, 1.0, 1.08883], dtype=np.float32)
    xyz = xyz * white
    M_inv = np.array([
        [3.2404542, -1.5371385, -0.4985314],
        [-0.9692660, 1.8760108, 0.0415560],
        [0.0556434, -0.2040259, 1.0572252],
    ], dtype=np.float32)
    lin = xyz @ M_inv.T
    mask = lin > 0.0031308
    rgb = np.where(mask, 1.055 * np.power(np.clip(lin, 0, None), 1 / 2.4) - 0.055, 12.92 * lin)
    return np.clip(rgb * 255, 0, 255)


def delta_e76(lab_a: np.ndarray, lab_b: np.ndarray) -> float:
    """LAB 공간 유클리드 거리 (CIE76). 방법론 비교 / palette chip 간 유사도 판정용."""
    return float(np.linalg.norm(np.asarray(lab_a) - np.asarray(lab_b)))


# --------------------------------------------------------------------------- #
# Skin 제거 (LAB box)
# --------------------------------------------------------------------------- #

def drop_skin(
    rgb_pixels: np.ndarray,
    lab_min: np.ndarray | None = None,
    lab_max: np.ndarray | None = None,
) -> np.ndarray:
    """LAB box 안에 드는 픽셀 제거. box 는 호출부에서 override 가능 (config 주입)."""
    if rgb_pixels.size == 0:
        return rgb_pixels
    lo = SKIN_LAB_MIN if lab_min is None else np.asarray(lab_min, dtype=np.float32)
    hi = SKIN_LAB_MAX if lab_max is None else np.asarray(lab_max, dtype=np.float32)
    lab = rgb_to_lab(rgb_pixels)
    inside = np.all((lab >= lo) & (lab <= hi), axis=-1)
    return rgb_pixels[~inside]


def hex_skin_leak(
    hex_code: str,
    lab_min: np.ndarray | None = None,
    lab_max: np.ndarray | None = None,
) -> bool:
    """추출된 dominant hex 가 skin LAB box 안인지. drop_skin 사후 QA 플래그."""
    lo = SKIN_LAB_MIN if lab_min is None else np.asarray(lab_min, dtype=np.float32)
    hi = SKIN_LAB_MAX if lab_max is None else np.asarray(lab_max, dtype=np.float32)
    lab = rgb_to_lab(hex_to_rgb(hex_code))
    return bool(np.all((lab >= lo) & (lab <= hi)))


# --------------------------------------------------------------------------- #
# KMeans 팔레트 추출 (LAB 공간)
# --------------------------------------------------------------------------- #

def extract_colors(
    rgb_pixels: np.ndarray,
    k: int = 5,
    min_pixels: int = 150,
) -> list[dict]:
    """RGB pixel 집합 → top-k cluster. 반환: {hex, weight, lab}[] weight desc sorted."""
    if rgb_pixels.shape[0] < min_pixels:
        return []
    lab = rgb_to_lab(rgb_pixels)
    k_eff = min(k, max(1, rgb_pixels.shape[0] // 50))
    km = KMeans(n_clusters=k_eff, n_init=4, random_state=0).fit(lab)
    counts = np.bincount(km.labels_, minlength=k_eff).astype(np.float32)
    weights = counts / counts.sum()
    centers_lab = km.cluster_centers_
    centers_rgb = lab_to_rgb(centers_lab)
    out: list[dict] = []
    for w, lab_c, rgb_c in zip(weights, centers_lab, centers_rgb):
        out.append({
            "hex": rgb_to_hex(rgb_c),
            "weight": float(w),
            "lab": lab_c.tolist(),
        })
    out.sort(key=lambda d: -d["weight"])
    return out
