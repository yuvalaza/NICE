"""Training procedure for NICE.
"""

import argparse
import torch
import torchvision
from torchvision import transforms
from collections import defaultdict
from tqdm import trange
import matplotlib.pyplot as plt
import nice


def train(flow, trainloader, optimizer, epoch):
    flow.train()
    train_loss = 0

    for n_batches, (inputs, _) in enumerate(trainloader, 1):
        inputs = inputs.view(inputs.shape[0], -1).to(next(flow.parameters()).device)

        optimizer.zero_grad()
        loss = -flow(inputs).mean()

        loss.backward()
        optimizer.step()

        train_loss += loss.item()


    total_loss = train_loss / len(trainloader)
    print(f"Epoch {epoch}: Train Loss = {total_loss:.4f}")
    return total_loss

def test(flow, testloader, filename, epoch, sample_shape):
    flow.eval()  # set to inference mode
    test_loss=0
    with torch.no_grad():
        samples = flow.sample(100).cpu()
        # a,b = samples.min(), samples.max()
        # samples = (samples-a)/(b-a+1e-10)
        
        samples = samples.view(-1,sample_shape[0],sample_shape[1],sample_shape[2])
        torchvision.utils.save_image(torchvision.utils.make_grid(samples),
                                     './samples/' + filename + 'epoch%d.png' % epoch)
        
        for n_batches, (inputs, _) in enumerate(testloader, 1):
            inputs = inputs.view(inputs.shape[0], -1).to(next(flow.parameters()).device)
            loss = -flow(inputs).mean()
            test_loss += loss.item()
            
    total_test_loss = test_loss / len(testloader)
    print(f"Epoch {epoch}: Test Log-Likelihood = {total_test_loss:.4f}")
    return total_test_loss

def main(args):
    mask_config=1
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    sample_shape = [1,28,28]
    full_dim =1*28*28
    transform  = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (1.,)),
        transforms.Lambda(lambda x: x + torch.zeros_like(x).uniform_(0., 1./256.)) #dequantization
    ])

    if args.dataset == 'mnist':
        trainset = torchvision.datasets.MNIST(root='./data/MNIST',
            train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset,
            batch_size=args.batch_size, shuffle=True, num_workers=2)
        testset = torchvision.datasets.MNIST(root='./data/MNIST',
            train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset,
            batch_size=args.batch_size, shuffle=False, num_workers=2)
    elif args.dataset == 'fashion-mnist':
        trainset = torchvision.datasets.FashionMNIST(root='~/torch/data/FashionMNIST',
            train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset,
            batch_size=args.batch_size, shuffle=True, num_workers=2)
        testset = torchvision.datasets.FashionMNIST(root='./data/FashionMNIST',
            train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset,
            batch_size=args.batch_size, shuffle=False, num_workers=2)
    else:
        raise ValueError('Dataset not implemented')

    model_save_filename = '%s_' % args.dataset \
             + 'batch%d_' % args.batch_size \
             + 'coupling%d_' % args.coupling \
             + 'coupling_type%s_' % args.coupling_type \
             + 'mid%d_' % args.mid_dim \
             + 'hidden%d_' % args.hidden \
             + '.pt'

    flow = nice.NICE(
                prior=args.prior,
                coupling=args.coupling,
                coupling_type=args.coupling_type,
                in_out_dim=full_dim, 
                mid_dim=args.mid_dim,
                hidden=args.hidden,
                mask_config=mask_config,
                device=device).to(device)

    optimizer = torch.optim.Adam(
        flow.parameters(),lr=args.lr, betas=(0.9, 0.9999))


    train_log_likelihoods = []
    test_log_likelihoods = []

    for epoch in range(1, args.epochs + 1):
        
        train_log_likelihood = train(flow, trainloader, optimizer, epoch)
        train_log_likelihoods.append(train_log_likelihood)

    
        test_log_likelihood = test(flow, testloader, model_save_filename, epoch, sample_shape)
        test_log_likelihoods.append(test_log_likelihood)


    plt.figure()
    plt.plot(range(1, args.epochs + 1), train_log_likelihoods, label='Train')
    plt.plot(range(1, args.epochs + 1), test_log_likelihoods, label='Test')
    plt.xlabel('Epoch')
    plt.ylabel('Log-Likelihood')
    plt.legend()
    plt.title(f'Log-Likelihood ({args.dataset}, {args.coupling_type} Coupling)')
    plt.savefig(f'./results/log_likelihood_{args.dataset}_{args.coupling_type}.png')
    plt.show()
    print(f"Log-likelihood plot saved to ./results/log_likelihood_{args.dataset}_{args.coupling_type}.png")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--dataset',
                        help='dataset to be modeled.',
                        type=str,
                        default='fashion-mnist')
    parser.add_argument('--prior',
                        help='latent distribution.',
                        type=str,
                        default='logistic')
    parser.add_argument('--batch_size',
                        help='number of images in a mini-batch.',
                        type=int,
                        default=128)
    parser.add_argument('--epochs',
                        help='maximum number of iterations.',
                        type=int,
                        default=50)
    parser.add_argument('--sample_size',
                        help='number of images to generate.',
                        type=int,
                        default=64)
    parser.add_argument('--coupling-type',
                        help='.',
                        type=str,
                        default='additive')
    parser.add_argument('--coupling',
                        help='.',
                        # type=int,
                        default=4)
    parser.add_argument('--mid-dim',
                        help='.',
                        type=int,
                        default=1000)
    parser.add_argument('--hidden',
                        help='.',
                        type=int,
                        default=5)
    parser.add_argument('--lr',
                        help='initial learning rate.',
                        type=float,
                        default=1e-3)

    args = parser.parse_args()
    main(args)
