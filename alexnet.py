from pathlib import Path
import torch
from torchvision.models import alexnet, AlexNet_Weights
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import kagglehub
import csv 


#for each network, export the reference dataset (divided among train and test), in the same format as the network would see it (e.g. after transforms), export the reference output for the net/layers for which moc has to be computed. 



weights = AlexNet_Weights.DEFAULT
transform = weights.transforms()
#model = alexnet(weights=weights)
un_model = alexnet(weights=None)
tr_model = alexnet(weights=weights)


comp = "imagenet-object-localization-challenge"
cache_root = Path.home() / ".cache" / "kagglehub" / "competitions" / comp

if not cache_root.exists():
    kagglehub.login()
    path = kagglehub.competition_download('imagenet-object-localization-challenge')
else:
    path = cache_root


train_dir = Path(path) / "ILSVRC" / "Data" / "CLS-LOC" / "train"
train_ds = ImageFolder(train_dir, transform=transform)
train_loader = DataLoader(
    train_ds,
    batch_size=1,
    shuffle=False,
    num_workers=0,
    pin_memory=False,
)



val_dir = Path(path) / "ILSVRC" / "Data" / "CLS-LOC" / "val"
val_ds = ImageFolder(val_dir, transform=transform)
val_loader = DataLoader(
    val_ds,
    batch_size=1,
    shuffle=False,
    num_workers=0,
    pin_memory=False,
)

device = torch.device("cpu")
un_model.eval()
tr_model.eval()
output_path = Path("data") / "imagenet"

def export_split_to_csv(loader, split_name):
    x_file = output_path / f"X_{split_name}.csv"
    y_file = output_path / f"Y_{split_name}.csv"
    tr_file = output_path / f"F_{split_name}.csv"
    un_file = output_path / f"F_un.csv"

    with (
        open(x_file, "w", newline="") as fx,
        open(y_file, "w", newline="") as fy,
        open(un_file, "w", newline="") as fun,
        open(tr_file, "w", newline="") as ftr,
    ):
        x_writer = csv.writer(fx)
        y_writer = csv.writer(fy)
        un_writer = csv.writer(fun)
        tr_writer = csv.writer(ftr)

        with torch.no_grad():
            for imgs, labels in loader:
                imgs = imgs.to(device)
                labels = labels.to(device)

                un_output = un_model(imgs)   # shape: [B, 1000]
                tr_output = tr_model(imgs)   # shape: [B, 1000]


                # flatten each image in the batch to a row
                x_flat = imgs[0].flatten().cpu().tolist()
                y_vec = torch.nn.functional.one_hot(labels, num_classes=un_output.shape[1]).float()
                y_row = y_vec[0].cpu().tolist()

                un_prob = torch.softmax(un_output, dim=1)
                tr_prob = torch.softmax(tr_output, dim=1)

                un_row = un_prob[0].cpu().tolist() # shape [1000]
                tr_row = tr_prob[0].cpu().tolist() # shape [1000]

                x_writer.writerow(x_flat)
                y_writer.writerow(y_row)
                tr_writer.writerow(tr_row)

                if split_name=="train":
                    un_writer.writerow(un_row)


    print(f"Exported {split_name} split to {output_path}")




#export_split_to_csv(train_loader, "train")

export_split_to_csv(val_loader, "test")


#for each model we store input, output on disk 





#add algo for per-layer bounds of cnn alexnet