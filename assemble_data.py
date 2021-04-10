from tqdm import tqdm
import torch
import sys

images, labels = torch.load(sys.argv[1])

fi = open("images.csv", "a")
fl = open("labels.csv", "a")

images = images.view(-1)
labels = labels.view(-1)

for i in tqdm(range(images.size(0))):
    fi.write(str(images[i].item())+",")

for l in tqdm(range(labels.size(0))):
    fl.write(str(labels[l].item())+",")

fi.close()
fl.close()
