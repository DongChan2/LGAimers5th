import torch.nn as nn 
import torch 
class CenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim, device):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.device = device

        # 각 클래스에 대한 중심 벡터를 초기화 (각 중심은 feature dimension을 가짐)
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim).to(device))

    def forward(self, features, labels):
        # 각 샘플의 라벨에 해당하는 중심을 선택하여 거리 계산
        batch_size = features.size(0)
        centers_batch = self.centers.index_select(0, labels.long())

        # L2 거리 계산
        loss = F.mse_loss(features, centers_batch)
        return loss