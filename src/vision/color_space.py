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

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans

# --------------------------------------------------------------------------- #
# Skin LAB box 기본값 (spec §4.1 ④ Plan B — Pipeline B 기본)
# --------------------------------------------------------------------------- #
# 2026-04-23 `scripts/collect_skin_lab.py` 실측 (120 post / 5.2M pixel) 기반 box C 채택:
# L p5-p95, a/b p2.5-p97.5. 이전 [40,10,15]→[80,25,35] 은 deep skin / 음영 skin 을 놓침.
# L 하단 p1~p5 (5-16) 구간은 segformer mis-seg 픽셀이 많아 제외 (p5=16.1 부터).
#
# 2026-04-23 재조정: L 상한 79.8 (p95) → 72.0 으로 하향.
# 이유: 흰옷/밝은 옷 음영 (L 72~80) 이 L 상한 내 → skin 으로 오판되어 drop.
# 72.0 ≈ L p85 근사 (mean 53.39 + 1.04*std 18.85). fair skin 얼굴 (L 80+) 은 face class
# mask 로 이미 잡혀 garment 추출 대상 아님.
# configs/local.yaml 의 `vision.skin_lab_box` 로 override 가능 — 본 모듈 기본값은 fallback.
SKIN_LAB_MIN: np.ndarray = np.array([16.1, 0.0, -2.6])
SKIN_LAB_MAX: np.ndarray = np.array([72.0, 29.6, 43.7])


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
    """LAB box 안에 드는 픽셀 제거. 무조건 bin 필터 — adaptive 보호 없음.

    대부분의 호출부는 `drop_skin_adaptive` 를 사용해야 한다. 이 함수는 테스트/디버깅 용.
    """
    if rgb_pixels.size == 0:
        return rgb_pixels
    lo = SKIN_LAB_MIN if lab_min is None else np.asarray(lab_min, dtype=np.float32)
    hi = SKIN_LAB_MAX if lab_max is None else np.asarray(lab_max, dtype=np.float32)
    lab = rgb_to_lab(rgb_pixels)
    inside = np.all((lab >= lo) & (lab <= hi), axis=-1)
    return rgb_pixels[~inside]


def drop_skin_adaptive(
    rgb_pixels: np.ndarray,
    lab_min: np.ndarray | None = None,
    lab_max: np.ndarray | None = None,
    keep_threshold_pct: float = 0.5,
    upper_ceiling_pct: float = 0.97,
) -> tuple[np.ndarray, float, bool]:
    """3단 분기 — segformer 오분류/edge leak/skin-tone 옷을 나누어 처리.

    로직:
      1. drop_ratio > upper_ceiling_pct (0.97 기본) — 거의 전 pixel 이 skin-box 안.
         실제 베이지 kurta 도 shadow/주름으로 97%+ 찍기는 매우 어려움. 이 정도 비율은
         segformer 가 팔/다리 같은 순수 skin 을 upper-clothes/pants 로 오분류했을 가능성이
         높다 → segment 통째 drop (빈 배열 반환).
      2. drop_ratio > keep_threshold_pct (0.5 기본) — skin-tone 옷 (베이지/탄 kurta) 보존.
         원본 pixel 전체 유지 (kept_whole=True).
      3. 그 이하 — edge noise (목/손목 경계 skin leak) 로 간주, box 안 pixel 만 제거.

    반환: (cleaned_pixels, drop_ratio, kept_whole)
      - drop_ratio: box 안 pixel 의 비율 (판정 근거)
      - kept_whole: True 면 원본 유지 (skin-tone garment 판정). 나머지 두 분기는 False.
    """
    if rgb_pixels.size == 0:
        return rgb_pixels, 0.0, False
    lo = SKIN_LAB_MIN if lab_min is None else np.asarray(lab_min, dtype=np.float32)
    hi = SKIN_LAB_MAX if lab_max is None else np.asarray(lab_max, dtype=np.float32)
    lab = rgb_to_lab(rgb_pixels)
    inside = np.all((lab >= lo) & (lab <= hi), axis=-1)
    drop_ratio = float(inside.sum()) / rgb_pixels.shape[0]
    if drop_ratio > upper_ceiling_pct:
        return np.empty((0, 3), dtype=rgb_pixels.dtype), drop_ratio, False
    if drop_ratio > keep_threshold_pct:
        return rgb_pixels, drop_ratio, True
    return rgb_pixels[~inside], drop_ratio, False


def drop_skin_adaptive_spatial(
    crop_rgb: np.ndarray,
    garment_mask: np.ndarray,
    skin_class_mask: np.ndarray,
    lab_min: np.ndarray | None = None,
    lab_max: np.ndarray | None = None,
    keep_threshold_pct: float = 0.5,
    upper_ceiling_pct: float = 0.97,
    skin_dilate_iterations: int = 4,
) -> tuple[np.ndarray, float, bool]:
    """Spatial-aware 3단 방어. `drop_skin_adaptive` 의 확장 — 공간 정보 활용.

    Args:
      crop_rgb: (H, W, 3) uint8 — BBOX crop 원본.
      garment_mask: (H, W) bool — 이 instance 의 의류 class pixel (e.g., seg==4).
      skin_class_mask: (H, W) bool — segformer skin class (face/arms/legs) OR.
      skin_dilate_iterations: skin_class_mask 를 몇 px dilate 해 "인접 zone" 으로 삼을지.
        0 이면 spatial 방어 비활성 — pixel-list 분기와 동일하게 모든 box-in pixel drop.

    로직:
      1. garment_mask 안에서 LAB box 안 pixel → inside_mask
      2. drop_ratio = inside_mask.sum() / garment_mask.sum()
      3. ratio > upper_ceiling → 전 garment drop (segment 자체가 mis-seg 된 케이스)
      4. ratio > threshold → 전체 보존 (skin-tone 옷, 베이지 kurta 등)
      5. 그 이하 → drop_zone = skin_class_mask 를 N px dilate. 실제 drop 은
                  inside_mask ∩ drop_zone. 옷 내부 패턴 (쉬폰/자수/음영) 보존.

    반환: (cleaned_pixels, drop_ratio, kept_whole) — `drop_skin_adaptive` 와 동일 형식.
      downstream (KMeans) 의 pixel-list API 호환.
    """
    if garment_mask.sum() == 0:
        return np.empty((0, 3), dtype=crop_rgb.dtype), 0.0, False
    lo = SKIN_LAB_MIN if lab_min is None else np.asarray(lab_min, dtype=np.float32)
    hi = SKIN_LAB_MAX if lab_max is None else np.asarray(lab_max, dtype=np.float32)
    garment_pixels = crop_rgb[garment_mask]
    lab = rgb_to_lab(garment_pixels)
    inside_flat = np.all((lab >= lo) & (lab <= hi), axis=-1)
    drop_ratio = float(inside_flat.sum()) / garment_pixels.shape[0]
    if drop_ratio > upper_ceiling_pct:
        return np.empty((0, 3), dtype=crop_rgb.dtype), drop_ratio, False
    if drop_ratio > keep_threshold_pct:
        return garment_pixels, drop_ratio, True
    if skin_dilate_iterations <= 0:
        # 공간 방어 비활성 — `drop_skin_adaptive` 와 동일하게 box-in pixel 전부 drop.
        return garment_pixels[~inside_flat], drop_ratio, False
    if not skin_class_mask.any():
        # skin class 부재 (의류-only 제품샷 / segformer 가 skin 을 놓친 crop) → drop 근거 없음.
        # 내부 패턴 보존을 위해 전체 유지. 목/손목 leak 잡기는 포기.
        return garment_pixels, drop_ratio, False
    # scipy.ndimage 는 vision extras — 이 모듈은 의존성 없게 유지하려 lazy import.
    from scipy.ndimage import binary_dilation

    skin_zone = binary_dilation(skin_class_mask, iterations=skin_dilate_iterations)
    inside_mask = np.zeros_like(garment_mask)
    inside_mask[garment_mask] = inside_flat
    drop_mask = inside_mask & skin_zone
    keep_mask = garment_mask & ~drop_mask
    return crop_rgb[keep_mask], drop_ratio, False


@dataclass(frozen=True)
class SkinDropConfig:
    """`drop_skin_2layer` 파라미터 번들 — LAB box + 3단 분기 threshold.

    numpy-free (tuple[float, float, float]) 로 core 계약 compatible. 기본값은 모듈 기본
    SKIN_LAB_MIN/MAX + secondary 0.5 + ceiling 0.97 — 이전 signature defaults 와 동일.
    settings.VisionConfig 로부터 조립은 호출부 책임 (`canonical_extractor._build_skin_drop_config`).
    """
    lab_min: tuple[float, float, float] = (16.1, 0.0, -2.6)
    lab_max: tuple[float, float, float] = (72.0, 29.6, 43.7)
    secondary_drop_threshold_pct: float = 0.5
    upper_ceiling_pct: float = 0.97


def drop_skin_2layer(
    crop_rgb: np.ndarray,
    garment_mask: np.ndarray,
    segformer_skin_mask: np.ndarray,
    config: SkinDropConfig | None = None,
) -> tuple[np.ndarray, int, int]:
    """Phase 3 재설계용 2-layer skin drop — primary semantic, secondary LAB box.

    `drop_skin_adaptive_spatial` (legacy) 는 LAB box 가 주 방어선이고 segformer skin mask 는
    dilate-zone 제한 보조. 재설계(roadmap M3.A Step D §5)는 반대 — segformer skin mask 를
    primary 로 직접 drop, LAB box 는 secondary. segformer semantic 이 조명/음영에 robust 하다는
    가정에 기반 (per-PoC verdict).

    Args:
      crop_rgb: (H, W, 3) uint8 — BBOX crop 원본.
      garment_mask: (H, W) bool — 이 canonical outfit 의 의류 class pixel (upper+lower OR).
      segformer_skin_mask: (H, W) bool — face/arms/legs semantic mask (SKIN_CLASS_IDS).
      config: SkinDropConfig — lab box + 분기 threshold. None 이면 기본값.

    로직:
      1. primary: effective_garment = garment_mask AND NOT segformer_skin_mask.
         segformer argmax 특성상 보통 두 mask 는 disjoint 라 0 drop 하지만, 경계 조건
         (dilation / sub-pixel upsample) 에서 약간의 overlap 가능 — primary 로 직접 제거.
      2. secondary (LAB box): effective_garment 의 RGB 픽셀을 LAB 변환 후 box 안 픽셀 비율
         계산. 3단 분기:
           - ratio > upper_ceiling → segment 통째 drop
           - ratio > threshold → LAB drop 생략, 전체 보존
           - 그 이하 → LAB box 안 픽셀만 drop (edge skin leak 제거)

    반환: (cleaned_pixels, primary_drop_count, secondary_drop_count)
      - cleaned_pixels: (N, 3) — KMeans 입력에 그대로 사용 가능.
      - primary_drop_count: primary 단계에서 제거된 픽셀 수 (진단/로깅).
      - secondary_drop_count: secondary(LAB) 단계에서 제거된 픽셀 수 (진단/로깅).
    """
    cfg = config or SkinDropConfig()
    if garment_mask.sum() == 0:
        return np.empty((0, 3), dtype=crop_rgb.dtype), 0, 0
    effective_garment = garment_mask & ~segformer_skin_mask
    primary_drop_count = int(garment_mask.sum() - effective_garment.sum())
    if effective_garment.sum() == 0:
        return np.empty((0, 3), dtype=crop_rgb.dtype), primary_drop_count, 0
    garment_pixels = crop_rgb[effective_garment]
    lo = np.asarray(cfg.lab_min, dtype=np.float32)
    hi = np.asarray(cfg.lab_max, dtype=np.float32)
    lab = rgb_to_lab(garment_pixels)
    inside = np.all((lab >= lo) & (lab <= hi), axis=-1)
    inside_count = int(inside.sum())
    total = garment_pixels.shape[0]
    drop_ratio = inside_count / total
    if drop_ratio > cfg.upper_ceiling_pct:
        # 거의 전부 skin-tone — segformer mis-seg 로 판단, 전체 drop. secondary_drop 에 포함.
        return np.empty((0, 3), dtype=crop_rgb.dtype), primary_drop_count, inside_count
    if drop_ratio > cfg.secondary_drop_threshold_pct:
        # skin-tone garment 보존 — LAB drop 하지 않음.
        return garment_pixels, primary_drop_count, 0
    cleaned = garment_pixels[~inside]
    return cleaned, primary_drop_count, inside_count


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
