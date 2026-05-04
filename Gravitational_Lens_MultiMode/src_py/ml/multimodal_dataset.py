import torch
from torch.utils.data import Dataset, DataLoader
import os

class GravitationalLensDataset(Dataset):
    def __init__(self, pt_path):
        # 캐시된 pt 파일로부터 텐서를 직접 로드하여 실시간 연산 제거
        checkpoint = torch.load(pt_path, weights_only=True)
        self.lc_data = checkpoint['lc_data']
        self.spatial_data = checkpoint['spatial_data']
        self.targets = checkpoint['targets']

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        # 메모리에 올라간 텐서를 인덱싱하여 즉시 반환 (매우 빠름)
        return self.lc_data[idx], self.spatial_data[idx], self.targets[idx]

def get_multimodal_loader(pt_path, batch_size=32, num_workers=0, pin_memory=False):
    dataset = GravitationalLensDataset(pt_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                      num_workers=num_workers, pin_memory=pin_memory)

if __name__ == "__main__":
    # 테스트 실행 (미리 preprocess_and_cache.py 를 실행해야 합니다)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    pt_file = os.path.join(base_dir, '../../data/cached_dataset.pt')
    if not os.path.exists(pt_file):
        print(f"⚠️ 캐시 파일을 찾을 수 없습니다. 먼저 preprocess_and_cache.py를 실행하세요: {pt_file}")
    else:
        loader = get_multimodal_loader(pt_file)
        lc, spatial, target = next(iter(loader))
        print(f"시계열 배치 크기: {lc.shape}")   # 예상: [32, 100, 2]
        print(f"공간 배치 크기: {spatial.shape}") # 예상: [32, 4]
        print(f"타겟 배치 크기: {target.shape}")  # 예상: [32, 2]
        print("✅ 캐시된 데이터 연동 완료!")
