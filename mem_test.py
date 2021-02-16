import pytorch_lightning as pl
import torch
import efficientnet_pytorch
#import ipdb


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

    def training_step(self, batch, batch_idx):
        #ipdb.set_trace()
        out = self(batch[0])
        loss = self.loss(out, batch[1])
        return super().training_step(*args, **kwargs)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)


def gpu_test():
    batch_size = 2

    dataset = Datatest()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, pin_memory=True, drop_last=True, num_workers=4)
    base_model = efficientnet_pytorch.EfficientNet.from_pretrained('efficientnet-b0')
    model = LitModel(base_model)

    for x, y in dataloader:
        print(x.shape)
        print(y.shape)
        break

    #res = model(x)
    #print(res)
    
    # trainer = pl.Trainer(max_steps=1)

    # trainer.fit(model, dataloader)

if __name__ == "__main__":
    gpu_test()