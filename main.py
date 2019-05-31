from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import pickle


class MimicNet(nn.Module):
    def __init__(self, args):
        super(MimicNet, self).__init__()
        try:
            self.activation = F.__dict__[args.activation]
        except KeyError:
            raise KeyError('torch functional module doesnt implement the specified activation')
        self._mimic = True
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = self.conv1(x)  # [batch x 20 x 24 x 24]
        conv_1 = x.view(-1, 20 * 24 * 24)
        x = self.activation(x)
        x = F.max_pool2d(x, 2, 2)
        x = self.conv2(x)  # [batch x 50 x 8 x 8]
        conv_2 = x.view(-1, 50 * 8 * 8)
        x = self.activation(x)
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = self.fc1(x)    # [batch x 500]
        fc_1 = x
        x = self.activation(x)
        out = self.fc2(x)  # [batch x 10]
        if self._mimic:
            return torch.cat([conv_1.view(-1, 20 * 24 * 24), conv_2.view(-1, 50 * 8 * 8), fc_1], dim=1), out
        else:
            return out

    def mimic(self):
        self._mimic = True

    def classify(self):
        self._mimic = False


class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        out = self.fc2(x)
        return out


val_accs = []
val_losses = []


def train_mimic(args, model, mimic_model, device, train_loader, test_loader, optimizer, epoch):
    model.train()
    model.mimic()
    losses = []
    for batch_idx, (data, _) in enumerate(train_loader):
        if batch_idx % args.test_freq == 0:
            model.classify()
            test(args, model, device, test_loader)
            model.mimic()
        data = data.to(device)
        optimizer.zero_grad()
        with torch.no_grad():
            target_activations, target_output = mimic_model(data)
        activations, output = model(data)
        # get output  loss
        if args.bce_loss:
            target_output = F.sigmoid(target_output)
            output_loss = F.binary_cross_entropy_with_logits(output, target_output)
        else:
            # manually compute cross entropy between distributions encoded in logits
            output = F.log_softmax(output)
            target_output = F.softmax(target_output)
            output_loss = -torch.sum(output * target_output) / output.shape[0]
        loss = output_loss
        if args.mimic_activations:
            # get activation loss
            if args.mse_loss:
                activation_loss = F.mse_loss(activations, target_activations)
            else:
                activations = F.sigmoid(activations)
                target_activations = F.sigmoid(target_activations)
                activation_loss = F.binary_cross_entropy(activations, target_activations)
            loss += activation_loss
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            losses += [loss.item()]
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
    model.classify()
    return losses


batch_idx = 0


def train(args, model, device, train_loader, test_loader, optimizer, epoch):
    global batch_idx
    model.train()
    losses = []
    for _, (data, target) in enumerate(train_loader):
        if args.test_freq is not None and batch_idx % args.test_freq == 0:
            test(args, model, device, test_loader)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        if args.bce_loss:
            new_target = torch.zeros((target.shape[0], 10))
            new_target[np.arange(target.shape[0]), target] = 1.
            loss = F.binary_cross_entropy_with_logits(output, new_target)
        else:
            loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            losses += [loss.item()]
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
        batch_idx += 1
    return losses


def test(args, model, device, test_loader):
    global val_accs, val_losses
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            output = F.log_softmax(output, dim=1)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_acc = 100. * correct / len(test_loader.dataset)

    val_accs += [test_acc]
    val_losses += [test_loss]

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), test_acc))

    model.train()
    return test_loss, test_acc


def save_file(args, model):
    # save to different path depending on args
    path = args.experiment_id + '/' + args.experiment_id + '.pt'
    torch.save(model.state_dict(), path)


def main():
    global val_accs, val_losses
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--experiment-id', type=str, default='0')
    # Mimic Settings
    parser.add_argument('--activation', type=str, choices=['relu', 'sigmoid'], default='relu')
    parser.add_argument('--mse-loss', action='store_true', help='Formulate MSE Loss on activations? Otherwise'
                                                                'loss is formulated from passing raw layer output'
                                                                'through a sigmoid')
    parser.add_argument('--baseline', action='store_false', dest='mimic')
    parser.add_argument('--only-mimic-output', action='store_false', dest='mimic_activations')
    parser.add_argument('--mimic-model', type=str, default=None,
                        help='The model to mimic output of. If none then train on ground truth')
    parser.add_argument('--bce-loss', action='store_true',
                        help='Whether to train output on BCE Loss (both for mimicking and classification)')
    # Statistics settings
    parser.add_argument('--test-freq', type=int, default=None, help="Frequency of validating training model")
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
    if args.mimic_model is not None:
        assert args.mimic, "Must use MimicModel for mimic training. Pass --mimic"

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

    if args.mimic:
        model = MimicNet(args).to(device)
    else:
        model = Net(args).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    model.classify()
    if args.mimic_model is not None:
        mimic_model = MimicNet(args).to(device)
        mimic_model.load_state_dict(torch.load(args.mimic_model))

    train_losses = []
    for epoch in range(1, args.epochs + 1):
        if args.mimic_model is not None:
            train_losses += train_mimic(args, model, mimic_model, device, train_loader, test_loader, optimizer, epoch)
        else:
            train_losses += train(args, model, device, train_loader, test_loader, optimizer, epoch)
        if args.test_freq is None:
            val_loss, val_accuracy = test(args, model, device, test_loader)
            val_losses += [val_loss]
            val_accs += [val_accuracy]

    with open('%s/%s-train-loss.pkl' % (args.experiment_id, args.experiment_id), 'wb+') as f:
        pickle.dump(train_losses, f)
    with open('%s/%s-val-accuracy.pkl' % (args.experiment_id, args.experiment_id), 'wb+') as f:
        pickle.dump(val_accs, f)
    with open('%s/%s-val-loss.pkl' % (args.experiment_id, args.experiment_id), 'wb+') as f:
        pickle.dump(val_losses, f)

    if args.save_model:
        save_file(args, model)


if __name__ == '__main__':
    main()