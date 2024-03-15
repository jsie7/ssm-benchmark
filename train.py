import argparse
import torch
from mamba_ssm import Mamba as MambaLayer
import wandb
from tqdm import tqdm
import yaml
from dataloaders import SequenceDataset

## MODEL DEF ##

class GLU(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, input_dim * 2)
    def forward(self, x):
        out = self.linear(x)
        return out[:, :, :x.shape[2]] * torch.sigmoid(out[:, :, x.shape[2]:])

class MambaBlock(torch.nn.Module):
    def __init__(self, hidden_dim, state_dim, conv_dim, expansion, dropout, glu, norm, prenorm):
        super().__init__()
        self.mamba = MambaLayer(d_model=hidden_dim, d_state=state_dim, d_conv=conv_dim, expand=expansion)
        if glu:
            self.glu = GLU(hidden_dim)
        else:
            self.glu = None
        self.activation = torch.nn.GELU()
        self.dropout = torch.nn.Dropout(dropout)
        if norm in ["layer"]:
            self.norm = torch.nn.LayerNorm(hidden_dim)
        elif norm in ["batch"]:
            raise RuntimeError("dimensions don't agree for batch norm to work")
            self.norm = torch.nn.BatchNorm1d(hidden_dim)
        self.prenorm = prenorm
    def forward(self, x):
        skip = x
        if self.prenorm:
            x = self.norm(x)
        x = self.mamba(x)
        x = self.dropout(self.activation(x))
        if self.glu is not None:
            x = self.glu(x)
        x = self.dropout(x)
        x = x + skip
        if not self.prenorm:
            x = self.norm(x)
        return x
    
class Mamba(torch.nn.Module):
    def __init__(self, num_blocks, input_dim, output_dim, hidden_dim, state_dim, conv_dim, expansion, dropout, glu, norm, prenorm, pooling="mean"):
        super().__init__()
        self.linear_encoder = torch.nn.Linear(input_dim, hidden_dim)
        self.blocks = torch.nn.Sequential(*[MambaBlock(hidden_dim, state_dim, conv_dim, expansion, dropout, glu, norm, prenorm) for _ in range(num_blocks)])
        self.linear_decoder = torch.nn.Linear(hidden_dim, output_dim)
        self.pooling = pooling
        self.softmax = torch.nn.LogSoftmax(dim=1)
    def forward(self, x):
        x = self.linear_encoder(x)
        x = self.blocks(x)
        if self.pooling in ["mean"]:
            x = torch.mean(x, dim=1)
        elif self.pooling in ["max"]:
            x = torch.max(x, dim=1)[0]
        elif self.pooling in ["last"]:
            x = x[:,-1,:]
        else:
            x = x # no pooling
        x = self.linear_decoder(x)
        return torch.softmax(x, dim=1)

## train loop ##

def train(seed, trainloader, testloader, num_epochs, learning_rate, wd, num_blocks, input_dim, output_dim, hidden_dim, state_dim, conv_dim, expansion, dropout, glu, norm, prenorm, pooling):
    torch.manual_seed(seed)
    device = "cuda"
    model = Mamba(num_blocks, input_dim, output_dim, hidden_dim, state_dim, conv_dim, expansion, dropout, glu, norm, prenorm, pooling).to(device)
    nr_params = sum(p.numel() for p in model.parameters())
    print("Nr. of parameters: {0}".format(nr_params))
    wandb.log({"params": nr_params})
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min = 5e-6)
    running_loss = 0.0
    for epoch in range(num_epochs):
        for X, y, _ in tqdm(trainloader):
            optimizer.zero_grad()
            X = X.to(device)
            y = y.to(device)
            y_hat = model(X)
            loss = torch.nn.functional.cross_entropy(y_hat, y)
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
        train_loss = running_loss/len(trainloader)
        print("Loss: {0:.3f}".format(train_loss))
        scheduler.step()

        model.eval()
        running_accuracy = 0.0
        with torch.no_grad():
            for X, y, _ in tqdm(trainloader):
                X = X.to(device)
                y = y.to(device)
                y_hat = model(X)
                accuracy = (y_hat.argmax(dim=1) == y).float().sum() / len(y)
                running_accuracy += accuracy
        train_acc = running_accuracy / len(trainloader)
        print("Train accuracy: {0:.4f}".format(train_acc))

        running_accuracy = 0.0
        running_loss = 0.0
        with torch.no_grad():
            for X, y, _ in tqdm(testloader):
                X = X.to(device)
                y = y.to(device)
                y_hat = model(X)
                loss = torch.nn.functional.cross_entropy(y_hat, y)
                running_loss += loss.item()
                accuracy = (y_hat.argmax(dim=1) == y).float().sum() / len(y)
                running_accuracy += accuracy
        test_loss = running_loss/len(testloader)
        test_acc = running_accuracy / len(testloader)
        print("Test accuracy: {0:.4f}\n".format(test_acc))

        wandb.log({"train acc": train_acc, "test acc": test_acc, "train loss": train_loss, "test loss": test_loss})
        model.train()

def split_train_val(train, val_split):
    train_len = int(len(train) * (1.0-val_split))
    train, val = torch.utils.data.random_split(
        train,
        (train_len, len(train) - train_len),
        generator=torch.Generator().manual_seed(42),
    )
    return train, val

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, default="cifar-10.yaml", help="experiment config file")
    config = parser.parse_args().config
    print("Using config {0}".format(config))

    # get GPU info
    if not torch.cuda.is_available():
        raise NotImplementedError("Cannot run on CPU!")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gpu_type = torch.cuda.get_device_name(0)
    print("Running on {0}".format(gpu_type))
    
    # get args
    with open("configs/"+config) as stream:
        try:
            args = yaml.safe_load(stream)            
        except yaml.YAMLError as exc:
            raise RuntimeError(exc)

    args["GPU"] = gpu_type
    
    print(yaml.dump(args))

    # start wandb logging
    wandb.login(key="58d1b0b4e77ad3dd9ebd08eb490255e83aa70bfe")
    wandb.init(
            entity="ssm-eth",
            project="lra-benchmark",
            config=args,
            job_type="train",
    )
    
    ## prepare dataset
    args["dataset"].pop("name") # remove logging name
    dataset = SequenceDataset.registry[args["dataset"]["_name_"]](**args["dataset"])
    dataset.setup()

    # Dataloaders
    trainloader = dataset.train_dataloader(batch_size=args["train"]["batch_size"], shuffle=True)
    testloader = dataset.test_dataloader(batch_size=args["train"]["batch_size"], shuffle=False)
    if type(testloader) is dict:
        testloader = testloader[None]
    
    train(
        seed=args["seed"],
        trainloader=trainloader,
        testloader=testloader,
        num_epochs=args["train"]["num_epochs"],
        learning_rate=args["train"]["lr"],
        wd=args["train"]["wd"],
        dropout=args["train"]["dropout"],
        num_blocks=args["model"]["num_blocks"],
        input_dim=args["model"]["input_dim"],
        output_dim=args["model"]["output_dim"],
        hidden_dim=args["model"]["hidden_dim"],
        state_dim=args["model"]["state_dim"],
        conv_dim=args["model"]["conv_dim"],
        expansion=args["model"]["expansion"],
        glu=args["model"]["glu"],
        norm=args["model"]["norm"],
        prenorm=args["model"]["prenorm"],
        pooling=args["model"]["pooling"]
    )

    wandb.finish()

