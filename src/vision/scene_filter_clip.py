"""CLIPSceneFilter — 2-stage zero-shot scene + demographic 판정 실 구현 (Phase 1).

Stage 1 (`accept`): frame 전체 embedding 으로 scene / woman / adult softmax 판정. argmax
완화 — threshold 기반 (stage1_female_min / stage1_adult_min). v2 (2026-04-25, adult-woman-
only 통합): man AND child signal 이 모두 stage2_mix_threshold 이상일 때만
stage=stage1_mix_needs_stage2 — BBOX 단위 재판정 지시. 둘 중 하나만 켜진 케이스 (성인 여성
+ 성인 남성 만, 성인 여성 + 아동 만) 는 stage1_pass — Gemini v0.6 프롬프트가 비-adult-
female 검출 제외 방어.

Stage 2 (`classify_persons`): YOLO person BBOX crop 별 CLIP forward → gender/age softmax
→ stage2_female_min / stage2_adult_min 충족 BBOX 만 keep. 작은 BBOX 는 embedding
불안정해 min_person_bbox_side_px 미만 → too_small drop.

top-level import 로 torch / transformers / PIL 사용. vision extras 필수 경로.
core 코드는 `vision.scene_filter` 의 Protocol 만 바라봄 — 이 모듈은 smoke / diagnostics
진입 지점에서만 lazy import (pipeline_b_extractor.load_models 참고).

성능:
- Stage 1 forward ~50-80ms on mps. Stage 2 는 bbox 당 추가 forward — IG 는 보통 1~2 person.
- text prompts 는 생성 시점 1회 인코딩해 상수로 저장. gender / age prompt 는 Stage 1/2
  공유해 stage2 추가 encoding 0.
"""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from settings import SceneFilterConfig
from vision.scene_filter import FilterVerdict, PersonVerdict


class CLIPSceneFilter:
    """SceneFilter 실 구현. load_clip_filter 로 생성하고 accept() / classify_persons() 로 판정."""

    def __init__(
        self,
        model: CLIPModel,
        processor: CLIPProcessor,
        cfg: SceneFilterConfig,
        device: str,
    ) -> None:
        self._model = model
        self._processor = processor
        self._cfg = cfg
        self._device = device
        self._scene_emb = self._encode_prompts(cfg.scene_prompts)
        self._gender_emb = self._encode_prompts(cfg.gender_prompts)
        self._age_emb = self._encode_prompts(cfg.age_prompts)

    def _encode_prompts(self, prompts: list[str]) -> torch.Tensor:
        """text prompts → projection 된 L2-normalized embedding.

        현재 transformers 버전의 CLIPModel.get_text_features 가 BaseModelOutputWithPooling
        을 반환하는 케이스가 있어 text_model + text_projection 을 직접 호출.
        """
        inputs = self._processor(text=prompts, return_tensors="pt", padding=True)
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.no_grad():
            text_outputs = self._model.text_model(**inputs)
            features = self._model.text_projection(text_outputs.pooler_output)
        return features / features.norm(dim=-1, keepdim=True)

    def _image_embedding(self, rgb: np.ndarray) -> torch.Tensor:
        pil = Image.fromarray(rgb)
        inputs = self._processor(images=pil, return_tensors="pt").to(self._device)
        with torch.no_grad():
            vision_outputs = self._model.vision_model(**inputs)
            emb = self._model.visual_projection(vision_outputs.pooler_output)
        return emb / emb.norm(dim=-1, keepdim=True)

    @staticmethod
    def _softmax_scores(
        img_emb: torch.Tensor, text_emb: torch.Tensor, prompts: list[str],
    ) -> dict[str, float]:
        # CLIP logit scale 은 zero-shot 관례 상 고정 100 (openai/clip 예시 동일).
        logits = (img_emb @ text_emb.T).squeeze(0) * 100.0
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        return {p: float(s) for p, s in zip(prompts, probs)}

    def accept(self, rgb: np.ndarray, frame_id: str) -> FilterVerdict:  # noqa: ARG002
        cfg = self._cfg
        img_emb = self._image_embedding(rgb)
        scene_scores = self._softmax_scores(img_emb, self._scene_emb, cfg.scene_prompts)
        gender_scores = self._softmax_scores(img_emb, self._gender_emb, cfg.gender_prompts)
        age_scores = self._softmax_scores(img_emb, self._age_emb, cfg.age_prompts)

        scene_list = list(scene_scores.values())
        gender_list = list(gender_scores.values())
        age_list = list(age_scores.values())
        scene_pass_score = scene_list[cfg.scene_pass_index]
        female_score = gender_list[cfg.female_index]
        male_score = gender_list[cfg.male_index]
        adult_score = age_list[cfg.adult_index]
        child_score = age_list[cfg.child_index]

        # Stage 1 rejects — scene off-fashion / 여성 signal 부재 / 성인 signal 부재.
        if scene_pass_score < cfg.scene_min_pass_score:
            return FilterVerdict(
                False, "scene_reject", "stage1_reject",
                scene_scores, gender_scores, age_scores,
            )
        if female_score < cfg.stage1_female_min:
            return FilterVerdict(
                False, "stage1_female_low", "stage1_reject",
                scene_scores, gender_scores, age_scores,
            )
        if adult_score < cfg.stage1_adult_min:
            return FilterVerdict(
                False, "stage1_adult_low", "stage1_reject",
                scene_scores, gender_scores, age_scores,
            )

        # Stage 1 pass. 4-way mix (성인 여성 + 성인 남성 + 아동) 모두 감지될 때만 stage2
        # 지시 — adult-woman-only 통합 (v2). 둘 중 하나만 켜진 케이스 (예: 성인 여성 +
        # 성인 남성) 는 BBOX 게이트 우회 → Gemini v0.6 프롬프트가 비-adult-female 제외 방어.
        has_mix = (
            male_score >= cfg.stage2_mix_threshold
            and child_score >= cfg.stage2_mix_threshold
        )
        stage = "stage1_mix_needs_stage2" if has_mix else "stage1_pass"
        return FilterVerdict(
            True, "ok", stage, scene_scores, gender_scores, age_scores,
        )

    def classify_persons(
        self,
        rgb: np.ndarray,
        bboxes: list[tuple[int, int, int, int]],
    ) -> list[PersonVerdict]:
        cfg = self._cfg
        h, w = rgb.shape[:2]
        out: list[PersonVerdict] = []
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            x1c, y1c = max(0, x1), max(0, y1)
            x2c, y2c = min(w, x2), min(h, y2)
            short_side = min(x2c - x1c, y2c - y1c)
            if short_side < cfg.min_person_bbox_side_px:
                out.append(PersonVerdict(False, "too_small", bbox))
                continue
            crop = rgb[y1c:y2c, x1c:x2c]
            img_emb = self._image_embedding(crop)
            gender_scores = self._softmax_scores(img_emb, self._gender_emb, cfg.gender_prompts)
            age_scores = self._softmax_scores(img_emb, self._age_emb, cfg.age_prompts)
            female_score = list(gender_scores.values())[cfg.female_index]
            adult_score = list(age_scores.values())[cfg.adult_index]
            if female_score < cfg.stage2_female_min:
                out.append(PersonVerdict(
                    False, "stage2_female_low", bbox, gender_scores, age_scores,
                ))
                continue
            if adult_score < cfg.stage2_adult_min:
                out.append(PersonVerdict(
                    False, "stage2_adult_low", bbox, gender_scores, age_scores,
                ))
                continue
            out.append(PersonVerdict(
                True, "ok", bbox, gender_scores, age_scores,
            ))
        return out


def load_clip_filter(cfg: SceneFilterConfig, device: str) -> CLIPSceneFilter:
    """CLIP 가중치 로드 + CLIPSceneFilter 생성. 첫 호출 시 ~600MB 자동 다운로드."""
    processor = CLIPProcessor.from_pretrained(cfg.model_id)
    model = CLIPModel.from_pretrained(cfg.model_id).to(device)
    model.eval()
    return CLIPSceneFilter(model=model, processor=processor, cfg=cfg, device=device)
