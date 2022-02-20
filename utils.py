import torch


class NoBaseline():
    def __init__(self):
        super().__init__()

    def update(self, loss: torch.Tensor) -> None:
        pass

    def predict(self, loss: torch.Tensor) -> torch.Tensor:
        return torch.zeros(1, device=loss.device)


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

    def predict(self, loss: torch.Tensor) -> torch.Tensor:
        if self.mean_baseline.device != loss.device:
            self.mean_baseline = self.mean_baseline.to(loss.device)
        return self.mean_baseline


def find_lengths(messages: torch.Tensor) -> torch.Tensor:
    max_k = messages.size(1)
    zero_mask = messages == 0
    lengths = max_k - (zero_mask.cumsum(dim=1) > 0).sum(dim=1)
    lengths.add_(1).clamp_(max=max_k)

    return lengths
