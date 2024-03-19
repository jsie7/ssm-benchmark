import argparse
import torch
from torch.nn.functional import softplus, gelu
from mamba_ssm import Mamba as MambaLayer
import wandb
from tqdm import tqdm
import yaml
from dataloaders import SequenceDataset
from accelerated_scan.warp import scan

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

class RMSNorm(torch.nn.Module):
    def __init__(self, *, dim):
        super().__init__()
        self.scale = dim**-0.5
        self.gamma = torch.nn.Parameter(torch.ones(dim))

    def forward(self, x):
        x = x / x.norm(p=2, dim=-1, keepdim=True)
        return self.gamma / self.scale * x


class Hawk(torch.nn.Module):
    def __init__(self, *, dim=1024, expansion_factor=1.5, kernel_size=4):
        super().__init__()
        hidden = int(dim * expansion_factor)
        self.input = torch.nn.Linear(dim, 2*hidden, bias=False)
        self.conv = torch.nn.Conv1d(in_channels=hidden, out_channels=hidden, bias=True,
                              kernel_size=kernel_size, groups=hidden, padding=kernel_size-1)
        self.gates = torch.nn.Linear(hidden, 2*hidden, bias=True)
        self.forget_base = torch.nn.Parameter(torch.linspace(-4.323, -9, hidden))
        self.output = torch.nn.Linear(hidden, dim, bias=False)
        self.alpha_log_scale = torch.nn.Parameter(-8 * torch.ones(1), requires_grad=False)

        with torch.no_grad():
            self.input.weight.normal_(std=dim**-0.5)
            self.gates.weight.normal_(std=hidden**-0.5)
            self.output.weight.normal_(std=hidden**-0.5)

    def forward(self, x):
        _N, T, _C = x.shape
        gate, x = self.input(x).chunk(2, dim=-1)
        x = self.conv(x.mT)[..., :T].mT

        # RG-LRU: linear recurrent unit with input-dependent gating
        forget, input = self.gates(x).chunk(2, dim=-1)
        alpha = (self.alpha_log_scale * softplus(self.forget_base) * forget.sigmoid()).exp()
        beta = (1 - alpha**2 + 1e-6).sqrt()
        x = beta * input.sigmoid() * x

        h = scan(alpha.mT.contiguous(), x.mT.contiguous()).mT
        x = self.output(gelu(gate) * h)
        return x


class GatedMLP(torch.nn.Module):
    def __init__(self, *, dim=1024, expansion_factor=2):
        super().__init__()
        hidden = int(dim * expansion_factor)
        self.grow = torch.nn.Linear(dim, 2 * hidden, bias=False)
        self.shrink = torch.nn.Linear(hidden, dim, bias=False)

        with torch.no_grad():
            self.grow.weight.normal_(std=dim**-0.5)
            self.shrink.weight.normal_(std=hidden**-0.5)

    def forward(self, x):
        gate, x = self.grow(x).chunk(2, dim=-1)
        x = gelu(gate) * x
        return self.shrink(x)

class Griffin(torch.nn.Module):
    def __init__(self, dim, expansion, gmlp_expansion, kernel_size):
        super().__init__()
        self.hawk_norm = RMSNorm(dim=dim)
        self.hawk = Hawk(dim=dim, expansion_factor=expansion, kernel_size=kernel_size)
        self.hawk_gmlp_norm = RMSNorm(dim=dim)
        self.hawk_gmlp = GatedMLP(dim=dim, expansion_factor=gmlp_expansion)

    def forward(self, x):
        x = x + self.hawk(self.hawk_norm(x))
        x = x + self.hawk_gmlp(self.hawk_gmlp_norm(x))
        return x
    
class GriffinBlock(torch.nn.Module):
    def __init__(self, num_blocks, input_dim, output_dim, hidden_dim, expansion, gmlp_expansion, kernel_size, pooling):
        super().__init__()
        self.linear_encoder = torch.nn.Linear(input_dim, hidden_dim)
        self.blocks = torch.nn.Sequential(*[Griffin(hidden_dim, expansion, gmlp_expansion, kernel_size) for _ in range(num_blocks)])
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

def train_mamba(seed, trainloader, testloader, num_epochs, learning_rate, wd, num_blocks, input_dim, output_dim, hidden_dim, state_dim, conv_dim, expansion, dropout, glu, norm, prenorm, pooling):
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
        scheduler.step(epoch)
        wandb.log({"lr": optimizer.param_groups[0]['lr']})

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

def train_griffin(seed, trainloader, testloader, num_epochs, learning_rate, wd, num_blocks, input_dim, output_dim, hidden_dim, expansion, gmlp_expansion, kernel_size, pooling):
    torch.manual_seed(seed)
    device = "cuda"
    model = GriffinBlock(num_blocks, input_dim, output_dim, hidden_dim, expansion, gmlp_expansion, kernel_size, pooling).to(device)
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
        scheduler.step(epoch)
        wandb.log({"lr": optimizer.param_groups[0]['lr']})

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
    
    if args["model"]["layer"] == "mamba":
        train_mamba(
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
    elif args["model"]["layer"] == "griffin":
        train_griffin(
            seed=args["seed"],
            trainloader=trainloader,
            testloader=testloader,
            num_epochs=args["train"]["num_epochs"],
            learning_rate=args["train"]["lr"],
            wd=args["train"]["wd"],
            num_blocks=args["model"]["num_blocks"],
            input_dim=args["model"]["input_dim"],
            output_dim=args["model"]["output_dim"],
            hidden_dim=args["model"]["hidden_dim"],
            expansion=args["model"]["expansion"],
            gmlp_expansion=args["model"]["gmlp_expansion"],
            kernel_size=args["model"]["kernel_size"],
            pooling=args["model"]["pooling"]
        )
    else:
        raise RuntimeError("{0} is not a valid model option".format(args["model"]["layer"]))

    wandb.finish()

