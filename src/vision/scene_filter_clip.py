"""CLIPSceneFilter — zero-shot scene + gender + age 판정 실 구현.

top-level import 로 torch / transformers / PIL 사용. vision extras 필수 경로.
core 코드는 `vision.scene_filter` 의 Protocol 만 바라봄 — 이 모듈은 smoke / diagnostics
진입 지점에서만 lazy import (pipeline_b_extractor.load_models 참고).

성능:
- 이미지 1장 forward ~50-80ms on mps (base CLIP ViT-B/32). drop 비율이 높으면 YOLO +
  segformer 비용을 절감 (pre-filter 효과).
- text prompts 는 생성 시점 1회 인코딩해 상수로 저장. image 마다 재인코딩 안 함.
"""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from settings import SceneFilterConfig
from vision.scene_filter import FilterVerdict


class CLIPSceneFilter:
    """SceneFilter 실 구현. load_clip_filter 로 생성하고 accept() 로 frame 당 판정."""

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
        # CLIP logit scale (학습 시 temperature 보정) — model.logit_scale.exp() 가 원칙이지만
        # zero-shot 용도로는 고정 100 이 일반 관례 (openai/clip 예시 동일).
        logits = (img_emb @ text_emb.T).squeeze(0) * 100.0
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        return {p: float(s) for p, s in zip(prompts, probs)}

    def accept(self, rgb: np.ndarray, frame_id: str) -> FilterVerdict:  # noqa: ARG002
        img_emb = self._image_embedding(rgb)
        scene_scores = self._softmax_scores(img_emb, self._scene_emb, self._cfg.scene_prompts)
        gender_scores = self._softmax_scores(img_emb, self._gender_emb, self._cfg.gender_prompts)
        age_scores = self._softmax_scores(img_emb, self._age_emb, self._cfg.age_prompts)

        scene_list = list(scene_scores.values())
        gender_list = list(gender_scores.values())
        age_list = list(age_scores.values())
        gender_idx = int(np.argmax(gender_list))
        age_idx = int(np.argmax(age_list))
        scene_pass_score = scene_list[self._cfg.scene_pass_index]

        # scene: pass prompt 의 score 만 확인 (argmax 조건 X). 야외 배경 / product-like 구도
        # 에서도 fashion 신호가 임계값 이상이면 살림.
        if scene_pass_score < self._cfg.scene_min_pass_score:
            return FilterVerdict(False, "scene_reject", scene_scores, gender_scores, age_scores)
        if gender_idx != self._cfg.gender_pass_index:
            return FilterVerdict(False, "gender_reject", scene_scores, gender_scores, age_scores)
        if age_idx != self._cfg.age_pass_index:
            return FilterVerdict(False, "age_reject", scene_scores, gender_scores, age_scores)
        return FilterVerdict(True, "ok", scene_scores, gender_scores, age_scores)


def load_clip_filter(cfg: SceneFilterConfig, device: str) -> CLIPSceneFilter:
    """CLIP 가중치 로드 + CLIPSceneFilter 생성. 첫 호출 시 ~600MB 자동 다운로드."""
    processor = CLIPProcessor.from_pretrained(cfg.model_id)
    model = CLIPModel.from_pretrained(cfg.model_id).to(device)
    model.eval()
    return CLIPSceneFilter(model=model, processor=processor, cfg=cfg, device=device)
