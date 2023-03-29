import torch
import os


DATA_DIR_GUESSWHAT = os.path.expanduser("~/data/guesswhat/")
DATA_DIR_IMAGENET = os.path.expanduser("~/data/imagenet/")

H5_IDS_KEY = "ids"

GUESSWHAT_MAX_NUM_OBJECTS = 10
RESNET_IMG_FEATS_DIM = 2048
ViT_IMG_FEATS_DIM = 768


class NoBaseline():
    def __init__(self):
        super().__init__()

    def update(self, loss: torch.Tensor) -> None:
        pass

    def predict(self, device) -> torch.Tensor:
        return torch.zeros(1, device=device)


class MeanBaseline():
    def __init__(self):
        super().__init__()

        self.mean_baseline = torch.zeros(1, requires_grad=False)
        self.n_points = 0.0

    def update(self, loss: torch.Tensor) -> None:
        self.n_points += 1
        if self.mean_baseline.device != loss.device:
            self.mean_baseline = self.mean_baseline.to(loss.device)

        self.mean_baseline += (
            loss.detach().mean().item() - self.mean_baseline
        ) / self.n_points

    def predict(self, device) -> torch.Tensor:
        if self.mean_baseline.device != device:
            self.mean_baseline = self.mean_baseline.to(device)
        return self.mean_baseline


def find_lengths(messages: torch.Tensor, stop_at_eos=True) -> torch.Tensor:
    max_k = messages.size(1)
    if stop_at_eos:
        zero_mask = messages == 0
        lengths = max_k - (zero_mask.cumsum(dim=1) > 0).sum(dim=1)
    else:
        lengths = (messages != 0).sum(dim=1)

    # Add 1 to avoid 0 length
    lengths.add_(1).clamp_(max=max_k)

    return lengths
