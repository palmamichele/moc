from pathlib import Path
import torch
import copy 
import csv
import time 
import sys 
import torchvision.transforms as transforms
import numpy as np 
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from utils import Tee, NeuralNet, accuracy_under_attack, box_clipping, pgd_m_x, pgd_moc, make_t_grid
from pgdmoc.pgd_moc import moc_regularizer

#PARS
num_experiments=1
epsilon = 0.3 #usually be 0.3 for MNIST, 8/255 for CIFAR-10, 0.5 for CIFAR-100
alpha = 0.01 #learning rate for PGD
num_iter = 15 #of PGD
num_classes = 10
num_epochs=10
num_restarts = 1
seed=0
batch_size=128
nbins=10 #for MOC
learning_rate = alpha
norm="L2" #on X space
clipping=None
lambda_moc = 0.5
#



def l2_distance(x,y):
    diff = x-y
    return diff.flatten(1).norm(p=2, dim=1)




torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed)
torch.manual_seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


data_path = Path("data")
out_dir = Path("experiments") / "MNIST"
out_dir.mkdir(parents=True, exist_ok=True)

log_file = open(out_dir/"log.txt", "w", encoding="utf-8")
sys.stdout = Tee(sys.__stdout__, log_file)
sys.stderr = Tee(sys.__stderr__, log_file)

def l2_distance(x,y):
    diff = x-y
    return diff.flatten(1).norm(p=2, dim=1)



#Load the training and test sets

# Transform the data to torch tensors
transform = transforms.Compose([
    transforms.ToTensor()
])

train_data = MNIST(root=str(data_path), train=True, download=True, transform=transform)
test_data = MNIST(root=str(data_path), train=False, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

X_train = train_data.data.float() 
X_train_flat = X_train.view(X_train.shape[0], -1)
t_values = make_t_grid(X=X_train_flat, norm=norm, nbins=nbins)


for i in range(num_experiments):

    vanilla_model = NeuralNet(hidden_layers=3, hidden_units=64)
    un_model = copy.deepcopy(vanilla_model)
    criterion = nn.CrossEntropyLoss(reduction="none")
    optimizer = optim.SGD(vanilla_model.parameters(), lr=learning_rate, momentum=0.0)

    vanilla_model.train()
    for epoch in range(num_epochs):
        loss = 0.0
        for images, labels in train_loader:

            optimizer.zero_grad()
            logits = vanilla_model(images)
            loss = criterion(logits, labels).mean()
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch+1}/{num_epochs}], 'f'Training Loss: {loss.item():.4f}')
    #store train loss of the vanilla model


    vanilla_model.eval()
    with torch.no_grad():
        
        test_correct = 0
        test_total = 0
        for images, labels in test_loader:
            probs = torch.softmax(vanilla_model(images), dim=1)
            preds = probs.argmax(dim=1)
            test_total += labels.size(0)
            test_correct += (preds == labels).sum().item()

        test_acc = test_correct / test_total
    print(f"clean test acc = {test_acc:.4f}")


    aua_test = accuracy_under_attack(
        vanilla_model,
        test_loader,
        criterion,
        clipping,
        norm,
        epsilon,
        alpha,
        num_iter
    )

    print(f"test accuracy under attack  = {aua_test:.4f}")
    #store test accuracy adv

    #compute the modulus of continuity of the network itself
    # moc, t_values = pgd_moc(
    # vanilla_model,
    # X,
    # Y, 
    # criterion, #d_Y as loss function (assuming it satisfies metric properties)
    # clipping,
    # norm, #L2, L1, Linf
    # t_values=None, #t_1,...,t_K
    # step_size=None, 
    # numiter=num_iter,
    # numrestarts=numrestarts,
    # nbins=nbins,
    # batch_size=batch_size)

    adv_model = copy.deepcopy(un_model)
    adv_model.train()
    optimizer = optim.SGD(adv_model.parameters(), lr=learning_rate, momentum=0.0)
    for epoch in range(num_epochs):
        loss = 0.0
        for images, labels in train_loader:

            _, deltas = pgd_m_x(adv_model,images, labels,epsilon,criterion,norm,clipping,num_iter,alpha)
            adv_images = images+deltas
    
            optimizer.zero_grad()
            outputs = adv_model(adv_images)
            loss = criterion(outputs, labels).mean()
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch+1}/{num_epochs}], 'f'Training Loss (adv): {loss.item():.4f}')

    adv_model.eval()
    with torch.no_grad():
        
        test_correct = 0
        test_total = 0
        for images, labels in test_loader:
            probs = torch.softmax(adv_model(images), dim=1)
            preds = probs.argmax(dim=1)
            test_total += labels.size(0)
            test_correct += (preds == labels).sum().item()

        test_acc = test_correct / test_total
    print(f"clean test acc (adv) = {test_acc:.4f}")


    aua_test = accuracy_under_attack(
        adv_model,
        test_loader,
        criterion,
        clipping,
        norm,
        epsilon,
        alpha,
        num_iter
    )

    print(f"test accuracy under attack (adv)  = {aua_test:.4f}")


    moc_model = copy.deepcopy(un_model)
    moc_model.train()

  
    # X_reference = torch.stack([
    #     train_data[i][0]
    #     for i in range(len(train_data))
    # ])
    optimizer = optim.SGD(moc_model.parameters(), lr=learning_rate, momentum=0.0)
    for epoch in range(num_epochs):

        total_clean_loss = 0.0
        total_moc_loss = 0.0

        for images, labels in train_loader:

            optimizer.zero_grad()

            outputs = moc_model(images)

            clean_loss = criterion(outputs,labels).mean()

            # clean_loss.backward()

            # clean_grad_norm = torch.sqrt(
            #     sum(
            #         p.grad.detach().pow(2).sum()
            #         for p in moc_model.parameters()
            #         if p.grad is not None
            #     )
            # )

            # print(clean_grad_norm)

            #can we speed this up?
            moc_loss, omega_values, t_grid = moc_regularizer(
                model=moc_model,
                X=images,          #might also be X or X_reference...
                dy_f=l2_distance,
                norm=norm,
                t_values=t_values,
                clipping_f=clipping,
                step_size=alpha,
                num_iter=num_iter,
                num_restarts=num_restarts,
                batch_size=batch_size,
            )

            # moc_loss.backward()
            # moc_grad_norm = torch.sqrt(
            #     sum(
            #         p.grad.detach().pow(2).sum()
            #         for p in moc_model.parameters()
            #         if p.grad is not None
            #     )
            # )

            # print(moc_grad_norm)

           
            loss = (clean_loss+ lambda_moc* moc_loss)

            loss.backward()

            optimizer.step()

            total_clean_loss += clean_loss.item()
            total_moc_loss += moc_loss.item()

        print(
            f"Epoch [{epoch + 1}/{num_epochs}] "
            # f"Clean Loss: "
            # f"{total_clean_loss / len(train_loader):.4f} "
            # f"MOC Loss: "
            # f"{total_moc_loss / len(train_loader):.4f}"
            f"{loss.item()}"
        )




