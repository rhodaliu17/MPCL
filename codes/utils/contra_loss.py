import torch
import torch.nn as nn
import torch.nn.functional as F

class PrototypeContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        
        super().__init__()
        self.temperature = temperature

    def forward(self, labeled_prototypes, unlabeled_prototypes):
        
        assert labeled_prototypes.is_cuda and unlabeled_prototypes.is_cuda
        n, k, c = labeled_prototypes.shape
        labeled_prototypes = F.normalize(labeled_prototypes, p=2, dim=2)
        unlabeled_prototypes = F.normalize(unlabeled_prototypes, p=2, dim=2)
        labeled_protos = labeled_prototypes.view(n * k, c)
        unlabeled_protos = unlabeled_prototypes.view(n * k, c)

        
        all_protos = torch.cat([labeled_protos, unlabeled_protos], dim=0)
        
        sim_matrix = torch.matmul(all_protos, all_protos.T) / self.temperature
        
        labels = torch.repeat_interleave(torch.arange(n, device=sim_matrix.device), k)
        labels = torch.cat([labels, labels]) 
        
        mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        mask = mask.fill_diagonal_(0)
        
        positive_mask = mask
        positive_sim = sim_matrix * positive_mask
        positive_sim = positive_sim[positive_mask.bool()].view(len(all_protos), -1)
        
        negative_mask = 1 - mask
        negative_sim = sim_matrix * negative_mask
        negative_sim = negative_sim[negative_mask.bool()].view(len(all_protos), -1)
        
        numerator = torch.exp(positive_sim).sum(dim=1)
        denominator = torch.exp(negative_sim).sum(dim=1)
        losses = -torch.log(numerator / (numerator + denominator + 1e-8))
        losses_avg = losses.mean()
        return losses_avg

class PrototypeAlignment(nn.Module):
    def __init__(self, n_classes, n_prototypes, feature_dim, temperature=0.07):
       
        super().__init__()
        self.contrastive_loss = PrototypeContrastiveLoss(temperature)
        
    def forward(self, labeled_features, unlabeled_features):
       
        return self.contrastive_loss(labeled_features, unlabeled_features)