# src_py/ml/regressor.py
# Phase 5 → Phase 9: 고도화된 ML 회귀 엔진
# - GradientBoosting / RandomForest(500) / Ridge / Ensemble 선택 가능
# - StandardScaler 자동 적용 (sklearn Pipeline)
# - 5-Fold Cross Validation + Holdout Test
# - MSE / MAE / R² / MAPE 4종 평가 메트릭
# - joblib 기반 모델 저장/로드
# - 앙상블 모드: RF + GBR + Ridge 예측 평균

import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from copy import deepcopy


def _mean_absolute_percentage_error(y_true, y_pred):
    """MAPE 계산 (sklearn에 없는 경우를 대비한 자체 구현)"""
    y_true = np.array(y_true, dtype=np.float64)
    y_pred = np.array(y_pred, dtype=np.float64)
    # 0에 가까운 값에 의한 발산 방지
    mask = np.abs(y_true) > 1e-8
    if mask.sum() == 0:
        return 0.0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0


class TargetAgnosticRegressor:
    """
    모드 무관(Target-Agnostic) ML 회귀 엔진.
    단일 타겟(1D)과 다중 타겟(2D+) 모두 자동 처리.
    """

    ALGORITHM_REGISTRY = {
        'gradient_boosting': lambda: GradientBoostingRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42,
        ),
        'random_forest': lambda: RandomForestRegressor(
            n_estimators=500,
            max_depth=None,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
        'ridge': lambda: Ridge(alpha=1.0),
        'ensemble': lambda: None,  # 앙상블은 _build_pipeline에서 특수 처리
    }

    def __init__(self, algorithm: str = 'gradient_boosting'):
        if algorithm not in self.ALGORITHM_REGISTRY:
            raise ValueError(
                f"지원하지 않는 알고리즘: '{algorithm}'. "
                f"선택 가능: {list(self.ALGORITHM_REGISTRY.keys())}"
            )
        self.algorithm_name = algorithm
        self._base_estimator = self.ALGORITHM_REGISTRY[algorithm]()
        self.pipeline = None  # train_and_evaluate 호출 시 생성
        self._is_multioutput = False

    def _build_pipeline(self, Y):
        """Y의 차원에 따라 단일/다중 타겟 파이프라인을 자동 구성"""
        is_multi = hasattr(Y, 'ndim') and Y.ndim == 2 and Y.shape[1] > 1
        self._is_multioutput = is_multi

        if self.algorithm_name == 'ensemble':
            # 앙상블은 별도 처리 — _build_ensemble_pipelines에서 구성
            self._build_ensemble_pipelines(is_multi)
            return

        base = self.ALGORITHM_REGISTRY[self.algorithm_name]()

        # Y가 2D이고 열이 2개 이상이면 MultiOutput 래핑
        if is_multi:
            regressor = MultiOutputRegressor(base)
        else:
            regressor = base

        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', regressor),
        ])

    def _build_ensemble_pipelines(self, is_multi):
        """RF + GBR + Ridge 앙상블 파이프라인 3개를 구성"""
        rf = RandomForestRegressor(
            n_estimators=500, max_depth=None, min_samples_leaf=2,
            random_state=42, n_jobs=-1,
        )
        gbr = GradientBoostingRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            subsample=0.8, random_state=42,
        )
        ridge = Ridge(alpha=1.0)

        models = {'random_forest': rf, 'gradient_boosting': gbr, 'ridge': ridge}
        self._ensemble_pipelines = {}

        for name, base in models.items():
            if is_multi:
                regressor = MultiOutputRegressor(base)
            else:
                regressor = base

            self._ensemble_pipelines[name] = Pipeline([
                ('scaler', StandardScaler()),
                ('regressor', regressor),
            ])

        # pipeline은 첫 번째(RF)를 대표로 지정 (CV 스코어링용)
        self.pipeline = self._ensemble_pipelines['random_forest']

    def train_and_evaluate(self, X, Y):
        """
        학습 + 평가 통합 메서드 (기존 API 호환).

        Returns:
            tuple: (pipeline, metrics_dict)
                - pipeline: 학습 완료된 sklearn Pipeline 객체
                - metrics_dict: {'mse', 'mae', 'r2', 'mape', 'cv_r2_mean', 'cv_r2_std'}
        """
        # numpy 변환 (pandas DataFrame/Series 대응)
        X_arr = np.array(X, dtype=np.float64)
        Y_arr = np.array(Y, dtype=np.float64)

        # 파이프라인 구성
        self._build_pipeline(Y_arr)

        # ── 1단계: 5-Fold Cross Validation ──
        print(f"\n{'='*60}")
        print(f"🧠 ML 학습 시작 | 알고리즘: {self.algorithm_name.upper()}")
        print(f"   피처 수: {X_arr.shape[1]} | 샘플 수: {X_arr.shape[0]}")
        print(f"{'='*60}")

        print("\n📊 [1/3] 5-Fold Cross Validation 수행 중...")
        cv_scores = cross_val_score(
            self.pipeline, X_arr, Y_arr,
            cv=5, scoring='r2', n_jobs=-1
        )
        cv_r2_mean = cv_scores.mean()
        cv_r2_std = cv_scores.std()
        print(f"   CV R² Scores: {[f'{s:.4f}' for s in cv_scores]}")
        print(f"   CV R² Mean ± Std: {cv_r2_mean:.4f} ± {cv_r2_std:.4f}")

        # ── 2단계: Holdout Test ──
        print("\n🔬 [2/3] Holdout Test (80/20 split)...")
        X_train, X_test, Y_train, Y_test = train_test_split(
            X_arr, Y_arr, test_size=0.2, random_state=42
        )

        if self.algorithm_name == 'ensemble':
            predictions = self._ensemble_fit_predict(X_train, Y_train, X_test)
        else:
            self.pipeline.fit(X_train, Y_train)
            predictions = self.pipeline.predict(X_test)

        # ── 3단계: 4종 메트릭 산출 ──
        mse = mean_squared_error(Y_test, predictions)
        mae = mean_absolute_error(Y_test, predictions)
        r2 = r2_score(Y_test, predictions)
        mape = _mean_absolute_percentage_error(Y_test, predictions)

        metrics = {
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'mape': mape,
            'cv_r2_mean': cv_r2_mean,
            'cv_r2_std': cv_r2_std,
        }

        print(f"\n📈 [3/3] Holdout Test 평가 결과:")
        print(f"   ├─ MSE  : {mse:.6e}")
        print(f"   ├─ MAE  : {mae:.6e}")
        print(f"   ├─ R²   : {r2:.4f}")
        print(f"   └─ MAPE : {mape:.2f}%")
        print(f"{'='*60}\n")

        # 전체 데이터로 최종 학습 (배포용)
        print("🔄 전체 데이터로 최종 모델 재학습 중...")
        if self.algorithm_name == 'ensemble':
            self._ensemble_fit_predict(X_arr, Y_arr, None)
        else:
            self.pipeline.fit(X_arr, Y_arr)
        print("✅ 최종 모델 학습 완료!\n")

        return self.pipeline, metrics

    def _ensemble_fit_predict(self, X_train, Y_train, X_test):
        """앙상블: RF + GBR + Ridge 각각 학습 후 예측 평균"""
        preds_list = []
        for name, pipe in self._ensemble_pipelines.items():
            pipe.fit(X_train, Y_train)
            if X_test is not None:
                pred = pipe.predict(X_test)
                r2 = r2_score(Y_train, pipe.predict(X_train))
                print(f"   [{name:20s}] Train R²: {r2:.4f}")
                preds_list.append(pred)

        if X_test is not None and preds_list:
            return np.mean(preds_list, axis=0)
        return None

    def predict(self, X):
        """학습된 모델로 추론 수행"""
        if self.pipeline is None:
            raise RuntimeError("모델이 학습되지 않았습니다. train_and_evaluate()를 먼저 호출하세요.")
        X_arr = np.array(X, dtype=np.float64)

        if self.algorithm_name == 'ensemble' and hasattr(self, '_ensemble_pipelines'):
            preds = [pipe.predict(X_arr) for pipe in self._ensemble_pipelines.values()]
            return np.mean(preds, axis=0)

        return self.pipeline.predict(X_arr)

    def save_model(self, filepath: str):
        """학습된 파이프라인을 joblib 파일로 저장"""
        if self.pipeline is None:
            raise RuntimeError("저장할 모델이 없습니다.")
        save_data = {
            'pipeline': self.pipeline,
            'algorithm': self.algorithm_name,
            'is_multioutput': self._is_multioutput,
        }
        if self.algorithm_name == 'ensemble' and hasattr(self, '_ensemble_pipelines'):
            save_data['ensemble_pipelines'] = self._ensemble_pipelines
        joblib.dump(save_data, filepath)
        print(f"💾 모델 저장 완료: {filepath}")

    def load_model(self, filepath: str):
        """저장된 모델을 로드"""
        data = joblib.load(filepath)
        self.pipeline = data['pipeline']
        self.algorithm_name = data['algorithm']
        self._is_multioutput = data['is_multioutput']
        if 'ensemble_pipelines' in data:
            self._ensemble_pipelines = data['ensemble_pipelines']
        print(f"📂 모델 로드 완료: {filepath} (알고리즘: {self.algorithm_name})")