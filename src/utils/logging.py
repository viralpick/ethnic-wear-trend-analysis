"""표준 logging 얇은 래퍼. 모듈 단위 get_logger만 제공한다."""
from __future__ import annotations

import logging

_DEFAULT_FORMAT = "[%(asctime)s] %(levelname)s %(name)s: %(message)s"


def get_logger(name: str) -> logging.Logger:
    # root handler가 없으면 한 번 설정. basicConfig 자체가 idempotent이지만 이중 설정
    # 로그를 피하려고 명시적으로 가드한다.
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format=_DEFAULT_FORMAT)
    return logging.getLogger(name)
