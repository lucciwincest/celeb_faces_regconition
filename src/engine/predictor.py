import torch


class Predictor:

    def __init__ (self, model, device):
        self.model = model.to (device)
        self.device = device

    @torch.no_grad ()
    def predict (self, X):
        self.model.eval ()
        X = X.to (self.device)
        output = self.model (X)
        return output.argmax (dim = 1)

    @torch.no_grad ()
    def predict_probability (self, X):
        self.model.eval ()
        X = X.to (self.device)
        output = self.model (X)
        return output.softmax (dim = 1)

    @torch.no_grad ()
    def predict_logits (self, X):
        self.model.eval ()
        X = X.to (self.device)
        return self.model (X)

    @torch.no_grad ()
    def test (self, test_loader):
        self.model.eval ()
        correct = 0
        total = 0

        for X, y in test_loader:
            X = X.to (self.device)
            y = y.to (self.device)
            pred = self.model (X).argmax (dim = 1)
            correct += (pred == y).sum ().item ()
            total += y.shape [0]

        acc = correct / total
        print (f"Test Accuracy: {acc:.4f}")
        return acc