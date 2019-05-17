from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import pickle


class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.activation = F.sigmoid
        self._mimic = True
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        conv_1 = self.activation(self.conv1(x))  # [batch x 20 x 24 x 24]
        x = F.max_pool2d(conv_1, 2, 2)
        conv_2 = self.activation(self.conv2(x))  # [batch x 50 x 8 x 8]
        x = F.max_pool2d(conv_2, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        fc_1 = self.activation(self.fc1(x))      # [batch x 500]
        out = self.fc2(fc_1)                     # [batch x 10]
        if self._mimic:
            return torch.cat([conv_1.view(-1, 20 * 24 * 24), conv_2.view(-1, 50 * 8 * 8), fc_1], dim=1), out
        else:
            return out

    def mimic(self):
        # Only called on mimicking model
        self._mimic = True

    def classify(self):
        # Only called on mimicking model
        self._mimic = False


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    model.classify()  # set model to only output logits
    losses = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        if args.bce_loss:
            new_target = torch.zeros((target.shape[0], 10))
            new_target[np.arange(target.shape[0]), target] = 1.
            loss = F.binary_cross_entropy_with_logits(output, new_target)
        else:
            #loss = F.cross_entropy(output, target)
            loss = F.nll_loss(F.log_softmax(output, dim=1), target, reduction='sum')  # sum up batch loss
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            losses += [loss.item()]
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
    return losses


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    return test_loss, 100. * correct / len(test_loader.dataset)


def save_file(args, model):
    # save to different path depending on args
    path = args.experiment_id
    path += '.pt'
    torch.save(model.state_dict(), path)


def main():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--experiment-id', type=str, default='0')
    # Mimic Settings
    parser.add_argument('--mimic', action='store_true')
    parser.add_argument('--bce-loss', action='store_true',
                        help='Whether to train output on BCE Loss (both for mimicking and classification)')
    # Training settings
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    model = Net(args).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    train_losses = []
    val_losses = []
    val_accuracies = []
    for epoch in range(1, args.epochs + 1):
        train_losses += train(args, model, device, train_loader, optimizer, epoch)
        val_loss, val_accuracy = test(args, model, device, test_loader)
        val_losses += [val_loss]
        val_accuracies += [val_accuracy]

    with open('%s-train-loss.pkl' % args.experiment_id, 'wb+') as f:
        pickle.dump(train_losses, f)
    with open('%s-val-accuracy.pkl' % args.experiment_id, 'wb+') as f:
        pickle.dump(val_accuracies, f)
    with open('%s-val-loss.pkl' % args.experiment_id, 'wb+') as f:
        pickle.dump(val_loss, f)

    if args.save_model:
        save_file(args, model)


if __name__ == '__main__':
    main()