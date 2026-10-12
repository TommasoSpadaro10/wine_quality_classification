import torch
import torch.nn as nn
import config

class CostSensitiveLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_classes = config.NUM_CLASSES
        self.cost_matrix = torch.zeros(self.num_classes, self.num_classes)
        
        for i in range(self.num_classes): # true class index
            for j in range(self.num_classes): # predicted class index
                true_quality = i + 1
                pred_quality = j + 1
                
                # Base penalty: distance between predicted and true quality
                penalty = float(abs(pred_quality - true_quality))
                
                # Extra penalties for cross-threshold errors
                true_is_good = true_quality > config.QUALITY_THRESHOLD
                pred_is_good = pred_quality > config.QUALITY_THRESHOLD
                
                if true_is_good and not pred_is_good:
                    # False Negative: Good wine predicted as Bad
                    penalty += config.PENALTY_GOOD_TO_BAD
                elif not true_is_good and pred_is_good:
                    # False Positive: Bad wine predicted as Good
                    penalty += config.PENALTY_BAD_TO_GOOD
                    
                self.cost_matrix[i, j] = penalty
                
        # Print the cost matrix before training
        print("Cost Matrix (True Class x Predicted Class):")
        print(self.cost_matrix)
        
        self.cost_matrix = self.cost_matrix.to(config.DEVICE)

    def forward(self, probs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Label smoothing parameter
        smooth = 0.1
        
        # Get the costs for the specific targets in the batch
        batch_costs = self.cost_matrix[target]
        
        # Calculate smoothed costs
        avg_costs = self.cost_matrix.mean(dim=0)
        smoothed_batch_costs = (1 - smooth) * batch_costs + smooth * avg_costs
        
        # Expected cost = sum(P(class_j) * SmoothedCost(target, class_j))
        expected_costs = torch.sum(probs * smoothed_batch_costs, dim=1)
        
        return expected_costs.mean()


class LatentCosineOrthogonalityLoss(nn.Module):
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, h1: torch.Tensor, h2: torch.Tensor) -> torch.Tensor:
        # Penalizza il quadrato della Cosine Similarity tra i vettori latenti
        cos_sim = torch.nn.functional.cosine_similarity(h1, h2, dim=1, eps=self.eps)
        return torch.mean(cos_sim ** 2)


class NegativeCorrelationLoss(nn.Module):
    """
    Negative Correlation Learning (NCL) Loss term.
    Calculates the covariance between the prediction residuals of Net 1 and Net 2:
    Cov(e1, e2) = mean_b [ sum_c (P1[b, c] - Y[b, c]) * (P2[b, c] - Y[b, c]) ]

    Minimizing Cov(e1, e2) explicitly penalizes correlated errors (both models failing together)
    and rewards negative correlation (complementary specializations).
    """
    def __init__(self, num_classes: int = config.NUM_CLASSES):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, p1: torch.Tensor, p2: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        y_onehot = torch.nn.functional.one_hot(target, num_classes=self.num_classes).float()
        res1 = p1 - y_onehot
        res2 = p2 - y_onehot
        covariance = torch.mean(torch.sum(res1 * res2, dim=1))
        return covariance
