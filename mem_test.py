import pytorch_lightning as pl
import torch
import efficientnet_pytorch


class Datatest(torch.utils.data.Dataset):
    def __getitem__(self, idx):
        return torch.rand((3, 224, 224)), torch.randint(low=0, high=1000, size=(1,))

    def __len__(self):
        return 3200

class LitModel(pl.LightningModule):
    def __init__(self, model):
        super().__init__()

        self.model = model
        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, x, y):
        out = self(x)
        loss = self.loss(out, x)
        return super().training_step(*args, **kwargs)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)


def gpu_test():
    batch_size = 1

    dataset = Datatest()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, pin_memory=True, drop_last=True, num_workers=4)
    base_model = efficientnet_pytorch.EfficientNet.from_pretrained('efficientnet-b0')
    model = LitModel(base_model)
    
    print(pl.Trainer.__doc__)
    trainer = pl.Trainer(max_steps=1)

    trainer.fit(model, dataloader)

if __name__ == "__main__":
    gpu_test()