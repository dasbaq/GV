import torch
import torch.nn as nn

# ---------------------------------------------------------
# 1. 시계열 데이터(광도 곡선) 특성 추출기 (Branch A)
# ---------------------------------------------------------
class LightcurveExtractor(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, num_layers=2):
        super(LightcurveExtractor, self).__init__()
        # input_dim: 2 (예: 관측 시간(Time), 밝기(Magnitude) 또는 오차(Error))
        # GRU(Gated Recurrent Unit)를 사용하여 시계열의 순차적 특징을 학습합니다.
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        
    def forward(self, x):
        # x shape: (Batch Size, Sequence Length, Input Dim)
        out, hidden = self.gru(x)
        # 마지막 시점의 은닉 상태(hidden state)만 추출하여 요약 벡터로 사용
        # hidden shape: (num_layers, Batch Size, hidden_dim)
        return hidden[-1] 

# ---------------------------------------------------------
# 2. 공간 데이터(Raytrace/이미지) 특성 추출기 (Branch B)
# ---------------------------------------------------------
class SpatialExtractor(nn.Module):
    def __init__(self, input_features=10, hidden_dim=64):
        super(SpatialExtractor, self).__init__()
        # 공간 데이터가 1D로 펼쳐진 좌표/특징 값이라고 가정 (MLP 구조)
        # 만약 픽셀 이미지(2D)라면 CNN(nn.Conv2d 등)으로 변경해야 합니다.
        self.network = nn.Sequential(
            nn.Linear(input_features, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, hidden_dim),
            nn.ReLU()
        )
        
    def forward(self, x):
        # x shape: (Batch Size, Input Features)
        return self.network(x)

# ---------------------------------------------------------
# 3. 데이터 융합 및 최종 예측 모델 (Late Fusion)
# ---------------------------------------------------------
class GravitationalLensMultiModal(nn.Module):
    def __init__(self, lc_input_dim=2, spatial_input_dim=10, output_dim=3):
        super(GravitationalLensMultiModal, self).__init__()
        
        # Branch A와 Branch B 초기화 (용량 증대: 64 -> 128)
        self.lc_branch = LightcurveExtractor(input_dim=lc_input_dim, hidden_dim=128, num_layers=3)
        self.spatial_branch = SpatialExtractor(input_features=spatial_input_dim, hidden_dim=128)
        
        # 융합된 벡터(128 + 128 = 256)를 받아 최종 파라미터를 예측하는 깊은 FCN
        self.fusion_layer = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3), # 과적합 방지
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, output_dim) # 최종 예측할 파라미터 개수
        )
        
    def forward(self, lc_data, spatial_data):
        # 1. 각각의 데이터에서 특징 추출
        lc_features = self.lc_branch(lc_data)
        spatial_features = self.spatial_branch(spatial_data)
        
        # 2. 두 특징 벡터를 하나로 연결 (Concatenate)
        # shape: (Batch Size, 128)
        combined_features = torch.cat((lc_features, spatial_features), dim=1)
        
        # 3. 최종 파라미터 예측
        predictions = self.fusion_layer(combined_features)
        return predictions

# ---------------------------------------------------------
# 4. 맞춤형 손실 함수 (Custom Loss Function)
# ---------------------------------------------------------
def custom_multimodal_loss(predictions, targets):
    """
    단순한 MSE를 넘어, 특정 물리적 도메인 지식을 반영할 수 있는 공간입니다.
    """
    criterion = nn.MSELoss()
    # 기본적으로는 예측값과 정답값(파라미터)의 평균 제곱 오차를 구합니다.
    loss = criterion(predictions, targets)
    
    # 향후 모델 고도화 시, 여기에 L1 정규화 패널티나
    # 물리 법칙을 위배하는 예측에 대한 추가 패널티(Physics-Informed Loss)를 더할 수 있습니다.
    return loss