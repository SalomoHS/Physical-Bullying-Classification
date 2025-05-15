import wandb
import torch
import torch.nn as nn
from torch import optim
import numpy as np
import timeit
from tqdm import tqdm

import model
from Bullying10k.braincog.utils import *
from Bullying10k.Bullying10k import get_bullying10k_data

def train(model, criterion, optimizer, data_loader):
    start_time = timeit.default_timer()
    running_loss = 0.0
    running_corrects = 0.0
    model.train()
    for inputs, labels in tqdm(data_loader['train']):
        inputs = inputs.to(device).float()
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)

        probs = nn.Softmax(dim=1)(outputs)
        preds = torch.max(probs, 1)[1]
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / trainval_sizes['train']
    epoch_acc = running_corrects.double() / trainval_sizes['train']
    wandb.log({"accuracy": epoch_acc, "loss": epoch_loss})
    print("[{}] Epoch: {}/{} Loss: {} Acc: {}".format('train', epoch+1, config.EPOCHS, epoch_loss, epoch_acc))
    stop_time = timeit.default_timer()
    print("Execution time: " + str(stop_time - start_time) + "\n")

def validate(model, criterion, optimizer, data_loader):
    start_time = timeit.default_timer()
    running_loss = 0.0
    running_corrects = 0.0

    all_preds = []
    all_labels = []
    model.eval()

    for inputs, labels in tqdm(data_loader['val']):
        inputs = inputs.to(device).float()
        labels = labels.to(device)
        optimizer.zero_grad()
        with torch.no_grad():
            outputs = model(inputs)

        probs = nn.Softmax(dim=1)(outputs)
        preds = torch.max(probs, 1)[1]
        loss = criterion(outputs, labels)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())


        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / trainval_sizes['val']
    epoch_acc = running_corrects.double() / trainval_sizes['val']
    wandb.sklearn.plot_confusion_matrix(all_labels, all_preds, [i for i in range(8)])
    print("[{}] Epoch: {}/{} Loss: {} Acc: {}".format('val', epoch+1, config.EPOCHS, epoch_loss, epoch_acc))
    stop_time = timeit.default_timer()
    print("Execution time: " + str(stop_time - start_time) + "\n")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wandb.login()

    wandb.init(
        project="<Project Name>",
        name="<Logging Name>",
    )

    config = wandb.config
    config.BATCH_SIZE = 4
    config.STEP = 30
    config.GAP = 4
    config.SIZE = 112
    config.NUM_CLASSES = 6
    config.LEARNING_RATE = 1e-5
    config.EPOCHS = 30
    config.SEED = 42
    config.WEIGHT_DECAY = 1e-4
    config.SIZE = 112
    config.LAYERS = (4, 4, 4, 4)
    config.CHANNELS =(64, 256, 512,1024)
    config.EXPANSION = 4
    
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)

    train_dataloader,test_dataloader,_,_ = get_bullying10k_data(batch_size = config.BATCH_SIZE, step = config.STEP,
                                                                gap = config.GAP, size = config.SIZE)
    val_dataloader = train_dataloader
    trainval_loaders = {'train': train_dataloader, 'val': val_dataloader}
    trainval_sizes = {x: len(trainval_loaders[x].dataset) for x in ['train', 'val']}
    test_size = len(test_dataloader.dataset)

    model = model.C3D(num_classes=config.NUM_CLASSES, pretrained=False, inchannel=2)
    # model = model.X3D(num_classes = config.NUM_CLASSES, layers = config.layers, 
    #                   channels=2, expansion = config.EXPANSION)
    # model = model.Inception3D(num_classes = config.NUM_CLASSES)
    
    model.to(device)
    criterion = nn.CrossEntropyLoss() 
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, 
                           weight_decay = config.WEIGHT_DECAY)

    wandb.watch(model, log="all")
    for epoch in range(config.EPOCHS):
        train(model,criterion,optimizer,trainval_loaders)
    
    for epoch in range(1):
        validate(model,criterion,optimizer,trainval_loaders)

    wandb.finish()
