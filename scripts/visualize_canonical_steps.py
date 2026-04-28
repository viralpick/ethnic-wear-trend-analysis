"""Canonical β-hybrid stage visualization.

Reads dump JSON produced by `scripts/diag_canonical_pool.py`
(`outputs/phase5_stepF_baseline/{post_id}.json`) and emits a stage-by-stage
color swatch + weight/share PNG.

Stages (per-canonical, per-object):
  Stage 0  raw  : KMeans clusters (pre-hybrid) + Gemini picks_input
  Stage 1  R3   : picks_after_R3 (Gemini picks survival vs KMeans deltaE76)
  Stage 2  R1+R2: every KMeans cluster categorized — R1 anchor / R1 merged
                  / R2 solo (vivid) / R2 merged (directional+hue rule) / R2 etc.
                  etc bucket weight rendered as hatched swatch.
  Stage 3  obj-final: weighted_clusters sorted by weight desc
  Stage 4  agg : flatten obj outputs -> weighted KMeans -> top_n cap
                 (cut_off_share recorded; etc weight folded in)

Usage:
  uv run python scripts/visualize_canonical_steps.py \
    --dump outputs/phase5_stepF_baseline/01KPT74FM28H0GT6MQTNHFBY1Q.json \
    --out  outputs/phase5_stepF_baseline/01KPT74FM28H0GT6MQTNHFBY1Q_steps.png
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
PRESET_PATH = REPO / "outputs" / "color_preset" / "color_preset.json"

# hybrid_palette.py 모듈 상수와 동일 (drift 시 pinning 테스트 실패).
# 시각화 replay 용 — settings.HybridPaletteConfig 와 같은 default.
CHROMA_VIVID = 15.0
HUE_NEAR_DEG = 30.0
R2_MERGE_DELTAE76 = 40.0
R2_MIN_SHARE = 0.10
PICK_MATCH_DELTAE76 = 25.0

ROLE_BORDER = {
    "R1 anchor": ("black", 2.5),
    "R1 merged": ("dimgray", 1.2),
    "R2 solo": ("green", 2.0),
    "R2 merged": ("steelblue", 1.6),
    "R2 etc": ("crimson", 1.6),
}


def _load_preset_lookup() -> dict[str, dict]:
    items = json.loads(PRESET_PATH.read_text(encoding="utf-8"))
    return {it["name"]: it for it in items}


def _swatch(ax, x, y, w, h, hex_color, label_top="", label_bottom="",
            border="black", border_w=1.0, top_fontsize=8, bottom_fontsize=7,
            crossed=False):
    rect = mpatches.Rectangle(
        (x, y), w, h, linewidth=border_w, edgecolor=border,
        facecolor=hex_color,
    )
    ax.add_patch(rect)
    if crossed:
        ax.plot([x, x + w], [y, y + h], color="red", linewidth=2)
        ax.plot([x, x + w], [y + h, y], color="red", linewidth=2)
    if label_top:
        ax.text(
            x + w / 2, y + h + 0.04, label_top,
            ha="center", va="bottom", fontsize=top_fontsize,
        )
    if label_bottom:
        ax.text(
            x + w / 2, y - 0.04, label_bottom,
            ha="center", va="top", fontsize=bottom_fontsize, family="monospace",
        )


def _deltaE76(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2) ** 0.5


def _resolve_pick_anchor(pick_lab, kmeans_clusters, threshold=25.0):
    """Closest cluster to pick. None if all over threshold."""
    best = None
    for c in kmeans_clusters:
        d = _deltaE76(pick_lab, c["lab"])
        if d > threshold:
            continue
        if best is None or d < best[1]:
            best = (c, d)
    return best


def _resolve_cluster_anchor(cluster_lab, surviving_picks_lab, threshold=25.0):
    """Cluster -> surviving pick anchor (unidirectional, min deltaE76)."""
    best = None
    for name, lab in surviving_picks_lab:
        d = _deltaE76(cluster_lab, lab)
        if d > threshold:
            continue
        if best is None or d < best[1]:
            best = (name, d)
    return best


def _chroma(lab):
    _, a, b = lab
    return (a * a + b * b) ** 0.5


def _hue_deg(lab):
    _, a, b = lab
    if (a * a + b * b) ** 0.5 < 1e-6:
        return None
    h = math.degrees(math.atan2(b, a))
    if h < 0:
        h += 360.0
    return h


def _hue_circular_diff(h1, h2):
    if h1 is None or h2 is None:
        return float("inf")
    d = abs(h1 - h2) % 360.0
    return min(d, 360.0 - d)


def _resolve_merge_target(cluster_lab, anchor_targets,
                          merge_threshold=R2_MERGE_DELTAE76,
                          hue_near_deg=HUE_NEAR_DEG):
    """hybrid_palette._resolve_merge_target replay — anchor_targets 는
    list of (anchor_name, target_lab). 통과 후보 중 ΔE76 최소 anchor 반환.
    """
    if not anchor_targets:
        return None
    c_chroma = _chroma(cluster_lab)
    c_hue = _hue_deg(cluster_lab)
    cL = cluster_lab[0]
    best = None
    for anchor_name, target_lab in anchor_targets:
        d = _deltaE76(cluster_lab, target_lab)
        if d > merge_threshold:
            continue
        t_hue = _hue_deg(target_lab)
        hue_diff = _hue_circular_diff(c_hue, t_hue)
        if hue_diff > hue_near_deg:
            t_chroma = _chroma(target_lab)
            tL = target_lab[0]
            if not (tL > cL and t_chroma > c_chroma):
                continue
        if best is None or d < best[1]:
            best = (anchor_name, d)
    return best[0] if best else None


def _categorize_clusters(kmeans, surviving_picks_lab):
    """build_object_palette 의 cluster fate 를 시각화용으로 replay.

    Returns:
      roles: list[dict] 길이 = len(kmeans). 각 dict 키:
        - role: {"R1 anchor", "R1 merged", "R2 solo", "R2 merged", "R2 etc"}
        - anchor_name: anchor pick name (R1*/R2 merged) or None
        - anchor_target_hex: L-highest target hex (R1 merged 가 흡수되는 좌표) or None
      anchor_targets: list[(anchor_name, target_lab, target_hex)]
    """
    by_anchor: dict[str, list] = {}
    for c in kmeans:
        anchor = _resolve_cluster_anchor(c["lab"], surviving_picks_lab,
                                         threshold=PICK_MATCH_DELTAE76)
        if anchor is None:
            continue
        by_anchor.setdefault(anchor[0], []).append(c)

    anchor_targets: list[tuple[str, tuple, str]] = []
    target_hex_by_anchor: dict[str, str] = {}
    target_lab_by_anchor: dict[str, tuple] = {}
    for name, _ in surviving_picks_lab:
        group = by_anchor.get(name)
        if not group:
            continue
        target = max(group, key=lambda c: c["lab"][0])
        anchor_targets.append((name, tuple(target["lab"]), target["hex"]))
        target_hex_by_anchor[name] = target["hex"]
        target_lab_by_anchor[name] = tuple(target["lab"])

    anchor_targets_lookup = [(n, lab) for n, lab, _h in anchor_targets]

    roles = []
    for c in kmeans:
        anchor = _resolve_cluster_anchor(c["lab"], surviving_picks_lab,
                                         threshold=PICK_MATCH_DELTAE76)
        if anchor is not None:
            name = anchor[0]
            target_hex = target_hex_by_anchor.get(name)
            if target_hex is not None and c["hex"] == target_hex:
                roles.append({"role": "R1 anchor", "anchor_name": name,
                              "anchor_target_hex": target_hex})
            else:
                roles.append({"role": "R1 merged", "anchor_name": name,
                              "anchor_target_hex": target_hex})
            continue
        c_chroma = _chroma(c["lab"])
        if c_chroma >= CHROMA_VIVID and c["share"] >= R2_MIN_SHARE:
            roles.append({"role": "R2 solo", "anchor_name": None,
                          "anchor_target_hex": None})
            continue
        target = _resolve_merge_target(c["lab"], anchor_targets_lookup)
        if target is not None:
            roles.append({"role": "R2 merged", "anchor_name": target,
                          "anchor_target_hex": target_hex_by_anchor.get(target)})
        else:
            roles.append({"role": "R2 etc", "anchor_name": None,
                          "anchor_target_hex": None})
    return roles, anchor_targets


def _draw_object_panel(fig, gs_row, obj, preset_lookup, obj_idx):
    axes = []
    titles = ["Stage 0 (raw)", "Stage 1 (R3 filter)",
              "Stage 2 (R1/R2 categorize + etc)", "Stage 3 (obj final)"]
    for col_idx, title in enumerate(titles):
        ax = fig.add_subplot(gs_row[col_idx])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_title(title, fontsize=10, pad=4)
        axes.append(ax)

    kmeans = obj["kmeans_clusters_pre_hybrid"]
    picks_input = obj["picks_input"]
    picks_r3 = obj["picks_after_R3"]
    weighted = obj["weighted_clusters"]

    pick_meta = []
    for name in picks_input:
        e = preset_lookup.get(name)
        if e is None:
            pick_meta.append({"name": name, "hex": "#888888",
                              "lab": (50, 0, 0), "missing": True})
        else:
            pick_meta.append({"name": name, "hex": e["hex"],
                              "lab": tuple(e["lab"])})

    # ---- Stage 0 ----
    ax = axes[0]
    ax.text(
        0.02, 0.96,
        f"obj {obj_idx} | pool={obj['obj_pixel_count']:,} px "
        f"| coverage={obj['frame_coverage_ratio']:.4f}",
        fontsize=7, family="monospace",
    )
    ax.text(0.02, 0.88, "KMeans (pre-hybrid):", fontsize=8, weight="bold")
    n = len(kmeans)
    sw_w = 0.85 / max(n, 1)
    for i, c in enumerate(kmeans):
        x = 0.07 + i * sw_w
        _swatch(
            ax, x, 0.55, sw_w * 0.8, 0.22, c["hex"],
            label_top=f"{c['hex']}",
            label_bottom=f"share\n{c['share']:.3f}",
            top_fontsize=7,
        )
        ax.text(
            x + sw_w * 0.4, 0.51,
            f"L{c['lab'][0]:.0f}\na{c['lab'][1]:.0f}\nb{c['lab'][2]:.0f}",
            ha="center", va="top", fontsize=6, family="monospace",
        )

    ax.text(0.02, 0.32, f"Gemini picks_input ({len(picks_input)}):",
            fontsize=8, weight="bold")
    for i, m in enumerate(pick_meta):
        x = 0.07 + i * sw_w
        _swatch(
            ax, x, 0.06, sw_w * 0.8, 0.18, m["hex"],
            label_top=m["name"],
            label_bottom=f"{m['hex']}",
            top_fontsize=7,
        )

    # ---- Stage 1 (R3) ----
    ax = axes[1]
    ax.text(0.02, 0.94, "R3 cutoff deltaE76 <= 25.0",
            fontsize=8, weight="bold")
    ax.text(
        0.02, 0.88,
        f"picks_after_R3 ({len(picks_r3)}/{len(picks_input)}):", fontsize=8,
    )
    surviving_picks_lab = []
    for i, m in enumerate(pick_meta):
        kept = m["name"] in picks_r3
        x = 0.07 + i * sw_w
        anchor = (_resolve_pick_anchor(m["lab"], kmeans)
                  if not m.get("missing") else None)
        deltae = anchor[1] if anchor else float("inf")
        anchor_hex = anchor[0]["hex"] if anchor else "-"
        label_top = m["name"]
        if anchor:
            label_bottom = f"{m['hex']}\ndE={deltae:.1f}\n->{anchor_hex}"
        else:
            label_bottom = f"{m['hex']}\ndE>25 DROP"
        _swatch(
            ax, x, 0.46, sw_w * 0.8, 0.30, m["hex"],
            label_top=label_top, label_bottom=label_bottom,
            top_fontsize=7, bottom_fontsize=6,
            crossed=not kept,
        )
        if kept:
            surviving_picks_lab.append((m["name"], m["lab"]))

    ax.text(
        0.02, 0.06,
        "R3: each pick -> closest KMeans cluster.\n"
        "<= 25 keep / > 25 hallucination drop",
        fontsize=7, family="monospace",
    )

    # ---- Stage 2 (R1 + R2 categorization + etc) ----
    ax = axes[2]
    ax.text(0.02, 0.96,
            "R1 anchor / R1 merged / R2 solo / R2 merged / R2 etc",
            fontsize=8, weight="bold")

    roles, anchor_targets = _categorize_clusters(kmeans, surviving_picks_lab)
    etc_weight = float(obj.get("r2_etc_weight", 0.0))

    # 한 행에 KMeans cluster + etc bucket 표시.
    n_slots = len(kmeans) + (1 if etc_weight > 0 else 0)
    sw_w_wc = 0.85 / max(n_slots, 1)
    for i, (c, role_info) in enumerate(zip(kmeans, roles)):
        x = 0.07 + i * sw_w_wc
        role = role_info["role"]
        anchor = role_info["anchor_name"]
        target_hex = role_info["anchor_target_hex"]
        border, border_w = ROLE_BORDER.get(role, ("black", 1.0))
        if role == "R1 anchor":
            label = f"R1 anchor\n<-{anchor}"
        elif role == "R1 merged":
            label = f"R1 merge\n->{anchor}\n({target_hex})"
        elif role == "R2 solo":
            label = "R2 solo\n(vivid)"
        elif role == "R2 merged":
            label = f"R2 merge\n->{anchor}\n({target_hex})"
        else:
            label = "R2 etc\n(drop->etc)"
        _swatch(
            ax, x, 0.40, sw_w_wc * 0.8, 0.32, c["hex"],
            label_top=c["hex"],
            label_bottom=f"share={c['share']:.3f}\n{label}",
            top_fontsize=7, bottom_fontsize=5,
            border=border, border_w=border_w,
        )
    if etc_weight > 0:
        x = 0.07 + len(kmeans) * sw_w_wc
        rect = mpatches.Rectangle(
            (x, 0.40), sw_w_wc * 0.8, 0.32,
            linewidth=1.6, edgecolor="crimson",
            facecolor="lightgray", hatch="////",
        )
        ax.add_patch(rect)
        ax.text(x + sw_w_wc * 0.4, 0.76, "etc bucket",
                ha="center", va="bottom", fontsize=7, weight="bold",
                color="crimson")
        ax.text(
            x + sw_w_wc * 0.4, 0.36,
            f"w={etc_weight:.1f}\n(R2 drops)",
            ha="center", va="top", fontsize=6, family="monospace",
        )

    weight_sum = sum(w["weight"] for w in weighted)
    ax.text(
        0.02, 0.06,
        f"weight = share * obj_pixel / frame_area * 10000\n"
        f"sum(weight)={weight_sum:.1f}  etc_weight={etc_weight:.1f}\n"
        f"R2: vivid+share>={R2_MIN_SHARE} keep / "
        f"merge dE76<={R2_MERGE_DELTAE76:.0f}, hue<={HUE_NEAR_DEG:.0f}deg",
        fontsize=6, family="monospace",
    )

    # ---- Stage 3 (object final) ----
    ax = axes[3]
    ax.text(0.02, 0.96, "Object output (weight desc)",
            fontsize=8, weight="bold")
    sorted_w = sorted(weighted, key=lambda w: -w["weight"])
    n_s = len(sorted_w)
    sw_w_sw = 0.85 / max(n_s, 1)
    for i, w in enumerate(sorted_w):
        x = 0.07 + i * sw_w_sw
        share_local = w["weight"] / weight_sum if weight_sum > 0 else 0
        _swatch(
            ax, x, 0.40, sw_w_sw * 0.8, 0.32, w["hex"],
            label_top=f"#{i+1} {w['hex']}",
            label_bottom=f"w={w['weight']:.1f}\nshare={share_local:.3f}",
            top_fontsize=7, bottom_fontsize=6,
        )

    return axes


def _draw_canonical_panel(fig, gs_row, canonical, all_objects):
    flat_input = []
    for obj in all_objects:
        for w in obj["weighted_clusters"]:
            flat_input.append(w)

    total_w = sum(w["weight"] for w in flat_input)

    ax_left = fig.add_subplot(gs_row[0:2])
    ax_left.set_xlim(0, 1)
    ax_left.set_ylim(0, 1)
    ax_left.axis("off")
    ax_left.set_title(
        f"Stage 4 input (aggregator) - flat size={len(flat_input)} "
        f"| sum(w)={total_w:.1f}",
        fontsize=10, pad=4,
    )

    n = len(flat_input)
    sw_w = 0.92 / max(n, 1)
    sorted_input = sorted(flat_input, key=lambda w: -w["weight"])
    for i, w in enumerate(sorted_input):
        x = 0.04 + i * sw_w
        share = w["weight"] / total_w if total_w > 0 else 0
        _swatch(
            ax_left, x, 0.40, sw_w * 0.8, 0.34, w["hex"],
            label_top=f"#{i+1} {w['hex']}",
            label_bottom=f"w={w['weight']:.1f}\nshare={share:.3f}",
            top_fontsize=7, bottom_fontsize=6,
        )

    ax_right = fig.add_subplot(gs_row[2:4])
    ax_right.set_xlim(0, 1)
    ax_right.set_ylim(0, 1)
    ax_right.axis("off")
    final_palette = canonical["palette"]
    cut_off = canonical["cut_off_share"]
    ax_right.set_title(
        f"Stage 4 output - top-3 cap | cut_off_share={cut_off:.4f}",
        fontsize=10, pad=4,
    )

    final_hexes = {p["hex"] for p in final_palette}
    n_final = len(final_palette)
    sw_w_f = 0.45 / max(n_final, 1)
    for i, p in enumerate(final_palette):
        x = 0.04 + i * sw_w_f
        _swatch(
            ax_right, x, 0.55, sw_w_f * 0.8, 0.30, p["hex"],
            label_top=f"#{i+1} {p['hex']}",
            label_bottom=f"share={p['share']:.4f}\n{p['family']}",
            top_fontsize=8, bottom_fontsize=7,
        )

    cut_clusters = [w for w in sorted_input if w["hex"] not in final_hexes]
    if cut_clusters:
        ax_right.text(
            0.55, 0.92, f"Cut by top_n=3 ({len(cut_clusters)}):",
            fontsize=8, weight="bold", color="darkred",
        )
        sw_w_c = 0.40 / max(len(cut_clusters), 1)
        for i, w in enumerate(cut_clusters):
            x = 0.55 + i * sw_w_c
            share = w["weight"] / total_w if total_w > 0 else 0
            _swatch(
                ax_right, x, 0.55, sw_w_c * 0.8, 0.30, w["hex"],
                label_top=f"{w['hex']}",
                label_bottom=f"w={w['weight']:.1f}\n({share:.3f})\nCUT",
                top_fontsize=7, bottom_fontsize=6,
                crossed=True,
            )

    note = (
        "Stage 4 = weighted KMeans (k = max obj_k) + weight desc "
        "+ top_n=3 cap -> renormalize -> family resolve\n"
        "cut_off_share = (top_n cap drop share) + (sum of obj r2_etc_weight share)"
    )
    ax_right.text(0.04, 0.10, note, fontsize=7, family="monospace")


def _draw(post_id, dump, out_path):
    preset_lookup = _load_preset_lookup()
    canonicals = dump["v3_hybrid"]["canonicals"]

    n_canonical = len(canonicals)
    if n_canonical != 1:
        print(f"warn: n_canonical={n_canonical}, "
              f"this script assumes single canonical")

    canonical = canonicals[0]
    objects = canonical["objects"]
    n_obj = len(objects)

    n_rows = n_obj + 2
    fig = plt.figure(figsize=(18, 3.5 * n_rows))
    gs = fig.add_gridspec(
        n_rows, 4,
        height_ratios=[0.4] + [1] * n_obj + [1.1],
        hspace=0.5, wspace=0.15,
    )

    ax_h = fig.add_subplot(gs[0, :])
    ax_h.axis("off")
    header_text = (
        f"Canonical beta-hybrid stage trace - post_id={post_id}\n"
        f"members={canonical['members_count']} "
        f"| objects={canonical['objects_count']} "
        f"| upper_ethnic={canonical['upper_is_ethnic']} "
        f"| lower_ethnic={canonical['lower_is_ethnic']} "
        f"| dress_as_single={canonical['dress_as_single']}"
    )
    ax_h.text(0.01, 0.6, header_text, fontsize=11, family="monospace",
              weight="bold")

    for obj_idx, obj in enumerate(objects):
        gs_row = gs[obj_idx + 1, :].subgridspec(1, 4, wspace=0.15)
        _draw_object_panel(fig, gs_row, obj, preset_lookup, obj_idx)

    gs_agg = gs[n_obj + 1, :].subgridspec(1, 4, wspace=0.15)
    _draw_canonical_panel(fig, gs_agg, canonical, objects)

    fig.suptitle(
        f"beta-hybrid trace - {post_id}",
        fontsize=14, weight="bold", y=0.995,
    )
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    dump = json.loads(args.dump.read_text(encoding="utf-8"))
    post_id = dump.get("post_id", args.dump.stem)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    _draw(post_id, dump, args.out)


if __name__ == "__main__":
    main()
