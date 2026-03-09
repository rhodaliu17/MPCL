import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# import numpy as np
from sklearn.cluster import KMeans

class ASGPrototype(nn.Module):
    def __init__(self, min_pixels=10, max_prototypes=4, mode = 'superpixel'):
        super().__init__()
        self.min_pixels = min_pixels
        self.max_prototypes = max_prototypes
        self.method = mode
        
    def getAdaptivePrototypes(self, img, fts, mask, region=False, method='Fast'):
        method = self.method
        num_classes = mask.shape[1]
        batch_size = mask.shape[0]
        img_norm = (img - img.min()) / (img.max() - img.min())
        spatial_coords = self.get_spatial_coords(fts.shape[2:])
        
        
        prototypes_list = []
        for c in range(num_classes):
            class_prototypes = []
            for b in range(batch_size):
                
                if torch.is_tensor(region):
                    curr_mask = mask[b,c,...] * region[b,...]
                    # curr_mask = mask[b,c,...]

                else:
                    curr_mask = mask[b,c,...]
                fts_with_coords = torch.cat([fts[b:b+1], spatial_coords.to(fts.device)], dim=1)
                protos = self.intensity_based_clustering_3d_histogram(
                        fts_with_coords.squeeze(0), 
                        curr_mask,
                        img_norm[b,0],
                        self.max_prototypes)
                    
                class_prototypes.extend(protos.unsqueeze(0))
            
            if len(class_prototypes) > 0:
                class_protos = torch.stack(class_prototypes, dim=0)
                avg_proto = torch.mean(class_protos, dim=0, keepdim=True)
                prototypes_list.append(avg_proto)
            else:
                zero_proto = torch.zeros(1, self.max_prototypes, fts.size(1)).to(fts.device)
                prototypes_list.append(zero_proto)


        return torch.cat([pro for pro in prototypes_list], dim=0)
    
    def get_spatial_coords(self, shape):
        z, y, x = shape
        coords_x = torch.linspace(-1, 1, x)
        coords_y = torch.linspace(-1, 1, y)
        coords_z = torch.linspace(-1, 1, z)
        
        coords = torch.stack(torch.meshgrid(coords_z, coords_y, coords_x))
        return coords[None]  # 1 x 3 x Z x Y x X

    def superpixel_clustering_3d(self, features, mask, num_clusters, n_iters=10, unlab=False):
        valid_mask = mask > 0.5
        
        if valid_mask.sum() == 0:
            return torch.zeros(num_clusters, features.size(0)-3).to(features.device)
            
        feat_masked = features[:, valid_mask]
        feat_masked = feat_masked.transpose(0, 1)
        
        perm = torch.randperm(feat_masked.size(0))
        sp_init_center = feat_masked[perm[:num_clusters]]
        sp_center = torch.zeros_like(sp_init_center).cuda()
        for i in range(n_iters):
            if i == 0:
                sp_center_rep = sp_init_center
            else:
                sp_center_rep = sp_center
            feat_dist = torch.cdist(
                feat_masked[:, :-3], 
                sp_center_rep[:, :-3]
            )
            spat_dist = torch.cdist(
                feat_masked[:, -3:], 
                sp_center_rep[:, -3:]
            )
            total_dist = torch.pow(feat_dist + spat_dist / 100, 0.5)
            p2sp_assoc = torch.neg(total_dist).exp()
            p2sp_assoc = p2sp_assoc / (p2sp_assoc.sum(0, keepdim=True))
            sp_center = feat_masked.unsqueeze(1) * p2sp_assoc.unsqueeze(-1) 
            sp_center = sp_center.sum(0)
            if i < n_iters - 1: 
                del feat_dist, spat_dist, total_dist
                torch.cuda.empty_cache()
       
        final_centers = sp_center[:, :-3]  # num_clusters x C
        if final_centers.size(0) < num_clusters:
            padding = torch.zeros(num_clusters - final_centers.size(0), final_centers.size(1)).to(final_centers.device)
            final_centers = torch.cat([final_centers, padding], dim=0)
        elif final_centers.size(0) > num_clusters:
            final_centers = final_centers[:num_clusters]
        del feat_masked, sp_init_center, sp_center_rep, sp_center
        return torch.cat([c.unsqueeze(0) for c in final_centers], dim=0) 

    def intensity_based_clustering_3d_histogram(self, features, mask, image, num_clusters, n_iters=10):
        valid_mask = mask > 0.5
        valid_count = valid_mask.sum().item()
        
        if valid_count == 0:
            return torch.zeros(num_clusters, features.size(0)-3).to(features.device)
        
        if valid_count < num_clusters:
            feat_masked = features[:, valid_mask]  # (C+3) x N
            feat = feat_masked[:-3, :]  # C x N
            
            repeat_times = (num_clusters + valid_count - 1) // valid_count
            feat_rep = feat.repeat(1, repeat_times)
            final_centers = feat_rep[:, :num_clusters].transpose(0, 1)
            return torch.cat([c.unsqueeze(0) for c in final_centers], dim=0)
        
        feat_masked = features[:, valid_mask]  # (C+3) x N
        feat_masked = feat_masked.transpose(0, 1)  # N x (C+3)
        
        intensity = image[valid_mask]  # N
        
        num_bins = max(100, num_clusters * 10)
        
        intensity_normalized = intensity
        bin_indices = (intensity_normalized * num_bins).long()
        bin_indices = torch.clamp(bin_indices, 0, num_bins - 1)
        
        hist_counts = torch.zeros(num_bins, dtype=torch.float32, device=intensity.device)
        ones = torch.ones_like(bin_indices, dtype=torch.float32)
        hist_counts.scatter_add_(0, bin_indices, ones)
        
        bin_edges = torch.linspace(0.0, 1.0, num_bins + 1, device=intensity.device)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        is_peak = (hist_counts > torch.roll(hist_counts, 1)) & (hist_counts > torch.roll(hist_counts, -1)) & (hist_counts > 0)
        
        is_peak[0] = False
        is_peak[-1] = False
        
        peak_indices = torch.nonzero(is_peak).squeeze(-1) 
        peak_counts = hist_counts[peak_indices] 
        
        if len(peak_indices) < num_clusters:
            top_k = min(num_clusters * 2, num_bins)
            top_bins_values, top_bins_indices = torch.topk(hist_counts, k=top_k)
            
            valid_mask_bins = top_bins_values > 0
            peak_indices = top_bins_indices[valid_mask_bins]
            peak_counts = top_bins_values[valid_mask_bins]
        
        sorted_indices = torch.argsort(peak_counts, descending=True)
        peak_indices = peak_indices[sorted_indices]
        peak_counts = peak_counts[sorted_indices]
        
        nms_threshold = 0.15
        
        selected_peaks = []
        for i in range(len(peak_indices)):
            peak_idx = peak_indices[i].item()
            peak_intensity = bin_centers[peak_idx].item()
            
            is_too_close = False
            for selected_idx in selected_peaks:
                selected_intensity = bin_centers[selected_idx].item()
                if abs(peak_intensity - selected_intensity) < nms_threshold:
                    is_too_close = True
                    break
            
            if not is_too_close:
                selected_peaks.append(peak_idx)
            
            if len(selected_peaks) >= num_clusters:
                break
        
        if len(selected_peaks) < num_clusters:
            for i in range(len(peak_indices)):
                peak_idx = peak_indices[i].item()
                if peak_idx not in selected_peaks:
                    selected_peaks.append(peak_idx)
                if len(selected_peaks) >= num_clusters:
                    break
        
        if len(selected_peaks) < num_clusters:
            all_indices = set(range(num_bins))
            remaining = list(all_indices - set(selected_peaks))
            selected_peaks.extend(remaining[:num_clusters - len(selected_peaks)])
        
        init_indices = []
        for peak_idx in selected_peaks[:num_clusters]:
            bin_min = bin_edges[peak_idx]
            bin_max = bin_edges[peak_idx + 1]
            
            in_bin_mask = (intensity >= bin_min) & (intensity < bin_max)
            bin_indices_list = torch.nonzero(in_bin_mask).squeeze(-1)
            
            if len(bin_indices_list) > 0:
                if bin_indices_list.dim() == 0:
                    init_indices.append(bin_indices_list.item())
                else:
                    random_idx = torch.randint(0, len(bin_indices_list), (1,)).item()
                    init_indices.append(bin_indices_list[random_idx].item())
            else:
                init_indices.append(torch.randint(0, len(intensity), (1,)).item())
        
        if len(init_indices) < num_clusters:
            remaining = list(set(range(valid_count)) - set(init_indices))
            if remaining:
                init_indices.extend(remaining[:num_clusters - len(init_indices)])
            else:
                while len(init_indices) < num_clusters:
                    init_indices.append(init_indices[0])
        
        sp_init_center = feat_masked[init_indices[:num_clusters]]
        sp_center = torch.zeros_like(sp_init_center).to(features.device)
        
        for i in range(n_iters):
            if i == 0:
                sp_center_rep = sp_init_center
            else:
                sp_center_rep = sp_center
            
            feat_dist = torch.cdist(feat_masked[:, :-3], sp_center_rep[:, :-3])
            spat_dist = torch.cdist(feat_masked[:, -3:], sp_center_rep[:, -3:])
            
            total_dist = torch.pow(feat_dist + spat_dist / 100, 0.5)
            p2sp_assoc = torch.neg(total_dist).exp()
            p2sp_assoc = p2sp_assoc / (p2sp_assoc.sum(1, keepdim=True) + 1e-10)
            
            sp_center = torch.zeros_like(sp_center_rep)
            for k in range(num_clusters):
                weights = p2sp_assoc[:, k].unsqueeze(1)
                weighted_sum = (feat_masked * weights).sum(0)
                weight_sum = weights.sum() + 1e-10
                sp_center[k] = weighted_sum / weight_sum
            
            if i < n_iters - 1:
                del feat_dist, spat_dist, total_dist, p2sp_assoc
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        final_centers = sp_center[:, :-3]
        
        if final_centers.size(0) < num_clusters:
            padding = torch.zeros(num_clusters - final_centers.size(0), 
                                final_centers.size(1)).to(final_centers.device)
            final_centers = torch.cat([final_centers, padding], dim=0)
        elif final_centers.size(0) > num_clusters:
            final_centers = final_centers[:num_clusters]
        
        return torch.cat([c.unsqueeze(0) for c in final_centers], dim=0)
    
    
    def calDist(self, fts, prototypes, eps=1e-8, scaler=1.0):
        
        N, C, X, Y, Z = fts.shape
        K = prototypes.size(0)
        
        fts_reshaped = fts.view(N, C, -1)  # [N, C, X*Y*Z]
        prototypes_reshaped = prototypes.view(K, C)  # [K, C]
        
        #[N, K, X*Y*Z]
        numerator = torch.einsum('nci,kc->nki', fts_reshaped, prototypes_reshaped)
        
        fts_norm = torch.norm(fts_reshaped, dim=1, keepdim=True)  # [N, 1,X*Y*Z]
        proto_norm = torch.norm(prototypes_reshaped, dim=1, keepdim=True)  # [K,1]
        
        norm_product = fts_norm * proto_norm.unsqueeze(0)  # [N, K, X*Y*Z]
        
        denominator = torch.clamp(norm_product, min=eps) 
        
        similarity = numerator / denominator  # [N, K, X*Y*Z]
        
        max_similarity, _ = torch.max(similarity, dim=1)  # [N, X*Y*Z]
        
        return max_similarity.view(N, X, Y, Z)