# -*- coding: utf-8 -*-
"""
설정 관리 모듈
임계값, 시퀀스 길이, 예측 시간대 등 설정
"""

import json
import os

# 설정 파일 경로
CONFIG_FILE = 'data/config.json'

# 기본 설정
DEFAULT_CONFIG = {
    "thresholds": {
        "1h": 0.3,      # 1시간: ±0.3%
        "24h": 1.5,     # 24시간: ±1.5%
        "7d": 5.0       # 7일: ±5.0%
    },
    "sequence_length": 72,  # 입력 시퀀스 길이 (72시간 = 3일)
    "prediction_horizons": {
        "1h": 1,
        "24h": 24,
        "7d": 168
    },
    "model_type": "lstm_xgboost",  # lstm_xgboost | svm
    "training": {
        "epochs": 50,
        "batch_size": 32,
        "validation_split": 0.2,
        "early_stopping_patience": 10
    },
    "features": {
        "numeric": ["open", "high", "low", "close", "volume"],
        "categorical": ["ma_cross", "rsi_signal", "volume_spike"]
    }
}


def load_config():
    """설정 파일 로드"""
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                config = json.load(f)
                # 기본값으로 없는 키 채우기
                return _merge_config(DEFAULT_CONFIG, config)
        except Exception as e:
            print(f"설정 로드 실패: {e}, 기본값 사용")
            return DEFAULT_CONFIG.copy()
    return DEFAULT_CONFIG.copy()


def save_config(config):
    """설정 파일 저장"""
    os.makedirs(os.path.dirname(CONFIG_FILE), exist_ok=True)
    with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"설정 저장 완료: {CONFIG_FILE}")


def _merge_config(default, custom):
    """기본 설정과 사용자 설정 병합"""
    result = default.copy()
    for key, value in custom.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _merge_config(result[key], value)
        else:
            result[key] = value
    return result


def get_threshold(config, horizon):
    """특정 시간대의 임계값 반환"""
    return config['thresholds'].get(horizon, 0.3)


def get_horizon_hours(config, horizon):
    """특정 시간대의 시간 수 반환"""
    return config['prediction_horizons'].get(horizon, 1)


def update_threshold(horizon, value):
    """임계값 업데이트"""
    config = load_config()
    config['thresholds'][horizon] = value
    save_config(config)
    return config


def get_all_horizons(config):
    """모든 예측 시간대 반환"""
    return list(config['prediction_horizons'].keys())


class ConfigManager:
    """설정 관리 클래스"""

    def __init__(self):
        self.config = load_config()

    def reload(self):
        """설정 다시 로드"""
        self.config = load_config()

    def save(self):
        """현재 설정 저장"""
        save_config(self.config)

    def get(self, key, default=None):
        """설정값 조회"""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
        return value if value is not None else default

    def set(self, key, value):
        """설정값 변경"""
        keys = key.split('.')
        config = self.config
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        config[keys[-1]] = value

    @property
    def thresholds(self):
        return self.config['thresholds']

    @property
    def sequence_length(self):
        return self.config['sequence_length']

    @property
    def horizons(self):
        return self.config['prediction_horizons']

    @property
    def model_type(self):
        return self.config['model_type']


# 전역 설정 인스턴스
config_manager = ConfigManager()


if __name__ == "__main__":
    # 테스트
    print("기본 설정:")
    print(json.dumps(DEFAULT_CONFIG, indent=2, ensure_ascii=False))

    print("\n설정 로드 테스트:")
    config = load_config()
    print(f"1시간 임계값: {get_threshold(config, '1h')}%")
    print(f"24시간 임계값: {get_threshold(config, '24h')}%")
    print(f"7일 임계값: {get_threshold(config, '7d')}%")
    print(f"시퀀스 길이: {config['sequence_length']}시간")
