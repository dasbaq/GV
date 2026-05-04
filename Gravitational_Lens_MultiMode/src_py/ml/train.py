import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
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
    # 전처리된 캐시 데이터(.pt) 경로 지정
    project_root = Path(__file__).resolve().parents[2]
    pt_path = project_root / "data" / "cached_dataset.pt"
    dataset = GravitationalLensDataset(pt_path=pt_path)
    
    train_loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda")
    )

    # 4. 모델 및 최적화 도구 초기화
    # lc_input_dim: 시간, 밝기(2) / spatial_input_dim: 좌표(4) / output: 파라미터(2)
    model = GravitationalLensMultiModal(lc_input_dim=2, spatial_input_dim=4, output_dim=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15)
    
    # 5. 학습 루프 시작
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (lc_data, spatial_data, targets) in enumerate(train_loader):
            # 첫 번째 에폭의 첫 번째 배치에서 스케일링(Standardization) 상태 디버깅 검증
            if epoch == 0 and i == 0:
                print("\n🔍 [데이터 검증] 첫 번째 배치 스케일링 상태:")
                print(f"   - 광도곡선(lc_data) 평균: {lc_data.mean().item():.4f}, 분산: {lc_data.var().item():.4f}")
                print(f"   - 공간정보(spatial_data) 평균: {spatial_data.mean().item():.4f}, 분산: {spatial_data.var().item():.4f}")
                print(f"   - 정답(targets) 평균: {targets.mean().item():.4f}, 분산: {targets.var().item():.4f}")
                
                print("\n🔍 [Sanity Check] 첫 번째 배치 샘플 확인:")
                print(f"   - 1번 샘플 타겟값: {targets[0].tolist()}")
                if len(targets) > 1:
                    print(f"   - 2번 샘플 타겟값: {targets[1].tolist()}")
                print(f"   - 1번 샘플 lc_data 100개 중 플럭스 합: {lc_data[0][:, 1].sum().item():.4f}")
                print(f"   - lc_data 전체가 0인가? {torch.all(lc_data == 0).item()}\n")

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
            
        epoch_loss = running_loss/len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}")
        
        # 스케줄러 업데이트
        scheduler.step(epoch_loss)

    # 6. 학습된 모델 저장
    torch.save(model.state_dict(), "gravitational_lens_model.pth")
    print("학습 완료 및 모델 저장 성공!")

if __name__ == "__main__":
    train()
