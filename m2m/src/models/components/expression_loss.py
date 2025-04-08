import torch
import torch.nn as nn

from .sdtw_cuda_loss import SoftDTW


class ExpressionL1Loss(nn.Module):
    def __init__(
        self, output_features, output_feature_boundaries, normalize=False, penalty_outliers=False
    ):
        super().__init__()
        self.l1_loss = nn.L1Loss(reduction="none")
        self.out_feat = output_features
        self.out_feat_bound = output_feature_boundaries
        self.normalize = normalize
        self.penalty = penalty_outliers

    def forward(self, x, y, mask):
        losses = {}
        for i in range(len(self.out_feat)):
            target = y[:, :, i].unsqueeze(-1)
            predict = x[i]
            # Only include the elements where mask is not zero
            valid_mask = mask.unsqueeze(-1) != 0
            # Filter both predict and target based on the mask
            valid_predict = predict[valid_mask]
            valid_target = target[valid_mask]

            if self.penalty:
                # Penalty on out of range values
                feat_bound = self.out_feat_bound[i]
                outliers = torch.sum(valid_predict < feat_bound[0]) + torch.sum(
                    valid_predict > feat_bound[1]
                )
                outliers = outliers / len(valid_predict) if self.normalize else outliers

            if len(valid_predict) and len(valid_target):
                # Compute l1_loss only on unmasked (valid) entries
                l1_loss = self.l1_loss(valid_predict, valid_target)
                if self.normalize:
                    l1_loss = torch.mean(l1_loss / valid_target)
                else:
                    l1_loss = torch.mean(l1_loss)

                if self.penalty:
                    l1_loss += 0.2 * outliers

            losses[f"{self.out_feat[i]}"] = {
                "l1_loss": l1_loss,
            }

        return losses


class ExpressionDWTLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.soft_dtw = SoftDTW(use_cuda=True, gamma=2, normalize=True, bandwidth=50)

    def forward(self, x, y):
        x = torch.stack(x, dim=2).squeeze()
        soft_dtw_loss = self.soft_dtw(x, y)
        return soft_dtw_loss


class ExpressionCrossEntropyLoss(nn.Module):
    def __init__(self, output_features, vocab_size):
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss(reduction="none")
        self.out_feat = output_features
        self.vocab_size = vocab_size

    def forward(self, x, y, mask):
        losses = {}
        vocabs = []
        for i in range(len(self.out_feat)):
            target = y[:, :, i]
            predict = x[i].permute(0, 2, 1)
            loss = self.cross_entropy(predict, target)
            loss = loss * mask
            loss = torch.sum(loss) / torch.sum(mask)

            losses[f"{self.out_feat[i]}"] = {
                "loss": loss.mean(),
            }

        for i in self.out_feat:
            vocabs.append(self.vocab_size[i])

        total_loss_all = [
            x * y for x, y in zip([losses[feat]["loss"] for feat in losses.keys()], vocabs)
        ]
        total_loss = sum(total_loss_all) / sum(vocabs)  # weighted

        return losses, total_loss
