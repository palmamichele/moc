from pathlib import Path
import torch
from torchvision.models import alexnet, AlexNet_Weights
from torchvision import transforms
from torch.utils.data import DataLoader, ConcatDataset
from torchvision.datasets import ImageFolder
from utils import lipschitz_from_fmoc, save_moc
import kagglehub
import numpy as np 
import time
import sys 
sys.path.append(str(Path("fmca") / "build" / "py"))
import FMCA



def minibatchmoc(loader, tr_model, un_model, TX,qX, nbins, norm, split_name, C):
    start = time.time()
    dmoc = FMCA.DiscreteModulusOfContinuity()
    dmoc.init(
        np.empty((1, 1), dtype=np.float64),   # dummy
        np.empty((1, 1), dtype=np.float64),   # dummy
        TX, qX, nbins, norm, norm
    )
    t_values = dmoc.tgrid()
    NT = len(t_values)

    #save on disk the very first grid, it will be the same for every other batch/split
    save_moc(t_values,save_path, f"batchdeltas_dmoc_{C}_{norm[0]}")

    final_moc_tr = np.zeros(NT)
    final_moc_un = np.zeros(NT)
    final_moc_data = np.zeros(NT)


    j=1
    for j, (imgs, labels) in enumerate(loader, start=1):
        imgs = imgs.to(device)

        print(f"batch i={j}/{len(loader)}")

        with torch.no_grad():
            F_trained = torch.softmax(tr_model(imgs), dim=1)
            F_untrained = torch.softmax(un_model(imgs), dim=1)

        F_trained = F_trained.cpu().numpy()
        F_untrained = F_untrained.cpu().numpy()
        P = imgs.view(imgs.size(0), -1).cpu().numpy()

        labels_np = labels.cpu().numpy()


        P = np.ascontiguousarray(P.T, dtype=np.float64) #points as columns 
        F_trained =  np.ascontiguousarray(F_trained.T, dtype=np.float64)  
        F_untrained = np.ascontiguousarray(F_untrained.T, dtype=np.float64)

        num_classes = len(F_untrained)
        F_data = np.eye(num_classes)[labels_np] 
        F_data = np.ascontiguousarray(F_data.T, dtype=np.float64)

        # --- trained ---
        dmoc_tr = FMCA.DiscreteModulusOfContinuity()
        dmoc_tr.init(P, F_trained, TX, qX, nbins, norm, norm)
        batch_tr = dmoc_tr.omegat()

        # --- untrained ---
        dmoc_un = FMCA.DiscreteModulusOfContinuity()
        dmoc_un.init(P, F_untrained, TX, qX, nbins, norm, norm)
        batch_un = dmoc_un.omegat()

        # --- data ---
        dmoc_data = FMCA.DiscreteModulusOfContinuity()
        dmoc_data.init(P, F_data, TX, qX, nbins, norm, norm)
        batch_data = dmoc_data.omegat()


        final_moc_tr = np.maximum(final_moc_tr, batch_tr)
        final_moc_un = np.maximum(final_moc_un, batch_un)
        final_moc_data = np.maximum(final_moc_data, batch_data)


    elapsed = time.time() - start
    save_moc(final_moc_tr, save_path, f"trained_batch{split_name}_dmoc_{C}_{norm[0]}")
    save_moc(final_moc_un, save_path, f"untrained_batch{split_name}_dmoc_{C}_{norm[0]}")
    save_moc(final_moc_data, save_path, f"data_batch{split_name}_dmoc_{C}_{norm[0]}")

    print(f"took {elapsed} s")



BATCH_SIZES = [10,100,1000]  #batch size that fits in memory
#compute u_b, l_b for TX, qX 
qX = 0.0001
TX = 1000
nbins =10000
device = "mps" #"cpu"

np.random.seed(0)
torch.manual_seed(0)

weights = AlexNet_Weights.DEFAULT
transform = weights.transforms()
#model = alexnet(weights=weights)
un_model = alexnet(weights=None).to(device)
tr_model = alexnet(weights=weights).to(device)
un_model.eval()
tr_model.eval()

comp = "imagenet-object-localization-challenge"
cache_root = Path.home() / ".cache" / "kagglehub" / "competitions" / comp

save_path = Path("experiments")/"imagenet"
save_path.mkdir(parents=True, exist_ok=True)

if not cache_root.exists():
    kagglehub.login()
    path = kagglehub.competition_download('imagenet-object-localization-challenge')
else:
    path = cache_root


train_dir = Path(path) / "ILSVRC" / "Data" / "CLS-LOC" / "train"
train_ds = ImageFolder(train_dir, transform=transform)

val_dir = Path(path) / "ILSVRC" / "Data" / "CLS-LOC" / "val"
val_ds = ImageFolder(val_dir, transform=transform)


norms = ["EUCLIDEAN","TAXICAB"] #["EUCLIDEAN", "TAXICAB"]
#conceptually   Pdata = dataList[b*C:b*C+C] and same for F 

for C in BATCH_SIZES:

    train_loader = DataLoader(
    train_ds,
    batch_size=C,
    shuffle=False,
    num_workers=0,
    pin_memory=False,
    drop_last=True
    )

    val_loader = DataLoader(
    val_ds,
    batch_size=C,
    shuffle=False,
    num_workers=0,
    pin_memory=False,
    drop_last=True
    )


    for norm in norms:
        minibatchmoc(val_loader, tr_model, un_model, TX,qX, nbins, norm, "test", C)
        minibatchmoc(train_loader, tr_model, un_model, TX,qX, nbins, norm, "train", C)
        # minibatchmoc(full_loader, tr_model, un_model, TX,qX, nbins, norm, "union", C) 
        for type in ["trained", "untrained", "data"]:
            #load train_moc, test_moc at saved location. (rmk: no cross batch interactions)
            train_moc =  np.loadtxt(save_path/f"{type}_batchtrain_dmoc_{C}_{norm[0]}.csv", delimiter=",")
            test_moc =  np.loadtxt(save_path/f"{type}_batchtest_dmoc_{C}_{norm[0]}.csv", delimiter=",")
            union_moc = np.maximum(test_moc, train_moc)
            save_moc(union_moc,save_path, f"{type}_batchunion_dmoc_{C}_{norm[0]}")










