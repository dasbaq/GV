import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from multimodal_model import GravitationalLensMultiModal, custom_multimodal_loss
from multimodal_dataset import GravitationalLensDataset

def train():
    # 1. 하이퍼파라미터 설정
    batch_size = 64
    epochs = 50
    learning_rate = 0.001
    
    # 2. 장치 설정 (맥북 전용 가속칩 'mps' 사용 확인)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Apple Silicon GPU(MPS) 모드로 학습을 시작합니다.")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        print("GPU를 찾을 수 없어 CPU 모드로 학습합니다.")

    # 3. 데이터 로더 준비
    # 실제 경로에 맞는 CSV 파일을 지정하세요.
    dataset = GravitationalLensDataset(spatial_csv='data/outputs/raytrace_mode_1.csv')
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 4. 모델 및 최적화 도구 초기화
    # lc_input_dim: 시간, 밝기(2) / spatial_input_dim: 좌표(4) / output: 파라미터(2)
    model = GravitationalLensMultiModal(lc_input_dim=2, spatial_input_dim=4, output_dim=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 5. 학습 루프 시작
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (lc_data, spatial_data, targets) in enumerate(train_loader):
            # 데이터를 장치(GPU)로 이동
            lc_data, spatial_data, targets = lc_data.to(device), spatial_data.to(device), targets.to(device)
            
            # 변화도(Gradient) 초기화
            optimizer.zero_grad()
            
            # 순전파 (Forward)
            outputs = model(lc_data, spatial_data)
            loss = custom_multimodal_loss(outputs, targets)
            
            # 역전파 (Backward) 및 가중치 업데이트
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}")

    # 6. 학습된 모델 저장
    torch.save(model.state_dict(), "gravitational_lens_model.pth")
    print("학습 완료 및 모델 저장 성공!")

if __name__ == "__main__":
    train()
