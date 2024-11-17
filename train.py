import argparse
import torch
import pytorch_warmup as warmup
import wandb
from tqdm import tqdm
import yaml
import sys
from dataloaders import SequenceDataset

from models import Mamba, Transformer
#, Hawk, SEaHawk

def train(seed, trainloader, testloader, model_cls,  wandb_config, train_config, model_config, checkpoint):
    torch.manual_seed(seed)
    device = "cuda"
    embedding = model_config["embedding"]
    model = model_cls(model_config).to(device)
    nr_params = sum(p.numel() for p in model.parameters())
    print("Nr. of parameters: {0}".format(nr_params))
    if wandb_config is not None:
        wandb.log({"params": nr_params})
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config["lr"], weight_decay=train_config["wd"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_config["num_epochs"], eta_min = 5e-6)
    warmup_scheduler = warmup.LinearWarmup(optimizer, train_config["warmup"])
    running_loss = 0.0
    for epoch in range(train_config["num_epochs"]):
        for X, y, _ in tqdm(trainloader):
            optimizer.zero_grad()
            if embedding:
                X = X.to(device)
            else:
                X = X.float()[:,:,None].to(device)
            y = y.to(device)
            y_hat = model(X)
            loss = torch.nn.functional.cross_entropy(y_hat, y)
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
        train_loss = running_loss/len(trainloader)
        print("Loss: {0:.3f}".format(train_loss))
        with warmup_scheduler.dampening():
            scheduler.step()

        model.eval()
        running_accuracy = 0.0
        with torch.no_grad():
            for X, y, _ in tqdm(trainloader):
                if embedding:
                    X = X.to(device)
                else:
                    X = X.float()[:,:,None].to(device)
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
                if embedding:
                    X = X.to(device)
                else:
                    X = X.float()[:,:,None].to(device)
                y = y.to(device)
                y_hat = model(X)
                loss = torch.nn.functional.cross_entropy(y_hat, y)
                running_loss += loss.item()
                accuracy = (y_hat.argmax(dim=1) == y).float().sum() / len(y)
                running_accuracy += accuracy
        test_loss = running_loss/len(testloader)
        test_acc = running_accuracy / len(testloader)
        print("Test accuracy: {0:.4f}\n".format(test_acc))

        if wandb_config is not None:
            wandb.log(
                {"train acc": train_acc,
                 "test acc": test_acc,
                 "train loss": train_loss,
                 "test loss": test_loss,
                 "lr": optimizer.param_groups[0]['lr']}
                )
        model.train()

    if checkpoint:
        torch.save(model.state_dict(), '/cluster/home/jsieber/projects/ssm-benchmark/checkpoint/ckpt.pth')

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
    print("\nUsing config {0}".format(config))

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
    checkpoint = args["save"]

    # get wandb config
    if "wandb" in args:
        wandb_config = args.pop("wandb")
    else:
        wandb_config = None
    
    print("\nCONFIG:")
    print(yaml.dump(args))

    # split configs
    data_config = args["dataset"]
    train_config = args["train"]
    model_config = args["model"]

    # start wandb logging
    if wandb_config is not None:
        wandb.login(key=wandb_config["key"])
        wandb.init(
                group=wandb_config["group"],
                name=wandb_config["name"],
                entity=wandb_config["entity"],
                project=wandb_config["project"],
                config=args,
                job_type="train",
        )
    
    # prepare dataset
    data_config.pop("name") # remove logging name
    dataset = SequenceDataset.registry[data_config["_name_"]](**data_config)
    dataset.setup()

    # dataloaders
    trainloader = dataset.train_dataloader(batch_size=train_config["batch_size"], shuffle=True)
    testloader = dataset.test_dataloader(batch_size=train_config["batch_size"], shuffle=False)
    if type(testloader) is dict:
        testloader = testloader[None]

    # extract model class [mamba | transformer | etc.]
    layer = model_config.pop("layer")
    
    # start train loop
    if layer == "mamba":
        model_cls = Mamba
    elif layer == "transformer":
        model_cls = Transformer
    # elif layer == "hawk":
    #     model_cls = Hawk
    # elif layer == "seahawk":
    #     model_cls = SEaHawk
    else:
        raise RuntimeError("{0} is not a valid model option".format(layer))
    
    train(
        args["seed"],
        trainloader,
        testloader,
        model_cls,
        wandb_config,
        train_config,
        model_config,
        checkpoint
    )
    
    try:
        if wandb_config is not None:
            wandb.finish()
    except:
        sys.exit(0)
