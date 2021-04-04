import torch
import sys

class MNIST(torch.nn.Module):

    def __init__(self):
        super(MNIST, self).__init__()

        self.linear = torch.nn.Sequential(
            torch.nn.Linear(784, 16),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Linear(16, 16),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Linear(16, 10),
            torch.nn.Softmax(dim=1)
        )

    def forward(self, input):
        return self.linear(input)

f = open("model.csv", "a")

model = MNIST()
model.load_state_dict(torch.load(sys.argv[1]))
param_layers = [0, 2, 4]
for l in param_layers:

    w_size = model.linear[l].weight.view(-1).size(0)
    b_size = model.linear[l].bias.view(-1).size(0)

    for w in range(w_size):
        f.write(str(model.linear[l].weight.view(-1)[w].item())+",")
    for b in range(b_size):
        f.write(str(model.linear[l].bias.view(-1)[b].item())+",")

f.close()
