import argparse
from functools import partial
import datetime
import time
import math
import sys
import os
import os.path
import numpy as np
from tqdm import tqdm
import gc

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
# noinspection PyPep8Naming
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.utils import save_image
import torchvision.datasets as vdsets

import cpflows.datasets as datasets
import cpflows.flows as flows
import cpflows.utils as utils
from cpflows.multiscale_flow import MultiscaleFlow
from cpflows.icnn import ICNN2, ConvICNN, ConvICNN2, ConvICNN3

torch.backends.cudnn.benchmark = True


def setup(rank, world_size, port):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size, timeout=datetime.timedelta(minutes=30))


def cleanup():
    dist.destroy_process_group()


def standard_normal_sample(size):
    return torch.randn(size)


# noinspection PyPep8Naming
def standard_normal_logprob(z):
    logZ = -0.5 * math.log(2 * math.pi)
    return logZ - z.pow(2) / 2


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def add_noise(x, nvals=256):
    """
    [0, 1] -> [0, nvals] -> add noise -> [0, 1]
    """
    noise = x.new().resize_as_(x).uniform_()
    x = x * (nvals - 1) + noise
    x = x / nvals
    return x


def identity(x):
    return x


def build_nnet(in_dim, h_dim, num_hidden_layers):
    nnet = []

    def conv3x3(d_in, d_out):
        return nn.Conv2d(d_in, d_out, kernel_size=3, padding=1)

    nnet.append(conv3x3(in_dim, h_dim))
    nnet.append(nn.ReLU(inplace=True))
    for _ in range(num_hidden_layers - 1):
        nnet.append(conv3x3(h_dim, h_dim))
        nnet.append(nn.ReLU(inplace=True))
    nnet.append(conv3x3(h_dim, in_dim * 2))
    return nn.Sequential(*nnet)


def cpflow_block_fn(index, input_shape, fc, block_type, dimh, num_hidden_layers, icnn_version, num_pooling):
    c, h, w = input_shape
    dim = c * h * w

    layers = []

    if block_type == "cvx":

        if fc:
            icnn = ICNN2(dim, dimh, num_hidden_layers)
        else:
            if icnn_version == 1:
                icnn = ConvICNN(c, dimh, num_hidden_layers, num_pooling=num_pooling)
            elif icnn_version == 2:
                icnn = ConvICNN2(c, dimh, num_hidden_layers)
            else:
                icnn = ConvICNN3(c, dimh, num_hidden_layers)

        layers.append(flows.DeepConvexFlow(icnn, dim, unbiased=False, no_bruteforce=True, rtol=1e-3, atol=1e-3))

    else:
        if input_shape[0] % 2 == 0:
            mask_type = f"channel{index % 2}"
        else:
            mask_type = f"checkerboard{index % 2}"
        nnet = build_nnet(input_shape[0], dimh, num_hidden_layers)
        layers.append(flows.MaskedCouplingBlock(input_shape[0], nnet, mask_type))

    return flows.SequentialFlow(layers)


# noinspection PyShadowingNames
def update_lr(optimizer, itr, args):
    iter_frac = min(float(itr + 1) / max(args.warmup_iters, 1), 1.0)
    lr = args.lr * iter_frac * args.ngpus
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def compute_loss(x, model):
    ndims = np.prod(x.shape[1:])
    nvals = 256  # for MNIST and CIFAR-10.

    z, logdet = model(x, 0)

    # log p(z)
    logpz = standard_normal_logprob(z).view(z.size(0), -1).sum(1)

    # log p(x)
    logpx = logpz + logdet - np.log(nvals) * ndims
    bits_per_dim = -torch.mean(logpx) / ndims / np.log(2)

    return bits_per_dim


# noinspection PyUnusedLocal
def reconstruction_loss(model, ema, data_loader, device):
    """Post-hoc experiment from https://github.com/asteroidhouse/INN-exploding-inverses/tree/master/ood-pretrained """

    if ema is not None:
        ema.swap()

    model.eval()

    transform = transforms.Compose([transforms.ToTensor()])

    minibatch_size = 25
    num_batches = 400

    tim_resize_data = vdsets.ImageFolder('ood_data/Imagenet_resize', transform=transform)
    tim_resize_data = np.stack([img.numpy() for (img, label) in tim_resize_data])

    tim_resized_recons_errs = []
    for batch_num in range(num_batches):
        print('tinyImageNet batch {}'.format(batch_num))
        tim_resize_minibatch = torch.from_numpy(
            tim_resize_data[batch_num * minibatch_size:batch_num * minibatch_size + minibatch_size]).float()
        # Range is [0, 1]
        tim_resize_minibatch = tim_resize_minibatch.cuda()

        with torch.no_grad():
            z, _ = model(tim_resize_minibatch.contiguous())
            recons = model(z, None, reverse=True)

        tim_resized_recons_errs += [torch.norm(x_hat - x).item() for (x_hat, x) in zip(recons, tim_resize_minibatch)]

    print('tinyImageNet-resized rec | mean: {} | min: {} | max: {}'.format(
        np.mean(tim_resized_recons_errs), np.min(tim_resized_recons_errs), np.max(tim_resized_recons_errs)))
    sys.stdout.flush()

    if ema is not None:
        ema.swap()


def rademacher_like(y):
    return torch.randint(low=0, high=2, size=y.shape).to(y) * 2 - 1


def visualize(model, x, fixed_z, savepath):
    z, _ = model(x)
    x_recon = model(z, None, reverse=True)
    x_samples = model(fixed_z, None, reverse=True)

    imgs = torch.cat([x, x_recon, x_samples], dim=0)
    save_image(imgs, savepath)


# noinspection PyUnusedLocal,PyShadowingNames
def train(epoch, train_loader, model, optimizer, bpd_meter, gnorm_meter, cg_meter, hnorm_meter, batch_time, ema, device,
          mprint, world_size, args):
    model.train()

    end = time.time()
    for i, (x, y) in enumerate(train_loader):

        global_itr = epoch * len(train_loader) + i
        update_lr(optimizer, global_itr, args)

        # Training procedure:
        # for each sample x:
        #   compute z = f(x)
        #   maximize log p(x) = log p(z) - log |det df/dx|

        x = x.to(device)

        # with torch.autograd.detect_anomaly():
        bpd = compute_loss(x, model)
        bpd_meter.update(bpd.item())

        loss = bpd
        loss.backward()

        grad_norm = torch.nn.utils.clip_grad.clip_grad_norm_(model.parameters(), 1.)

        optimizer.step()
        optimizer.zero_grad()
        ema.apply(decay=0.0 if epoch < 50 else None)

        gnorm_meter.update(grad_norm)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # keep track of number of CG iterates
        cg_meter.update(sum(flows.CG_ITERS_TRACER))
        flows.CG_ITERS_TRACER.clear()

        # keep track of Hessian norm
        if args.block_type == "cvx":
            hnorm_meter.update(sum(flows.HESS_NORM_TRACER) / len(flows.HESS_NORM_TRACER))
            flows.HESS_NORM_TRACER.clear()

        if i % args.logfreq == 0:
            s = (
                'Epoch: [{0}][{1}/{2}] | Time {batch_time.val:.3f} | '
                'GradNorm {gnorm_meter.avg:.2f} | CG iters {cg_meter.avg:.2f} | HNorm {hnorm_meter.avg:.3f}'.format(
                    epoch, i, len(train_loader), batch_time=batch_time, gnorm_meter=gnorm_meter, cg_meter=cg_meter,
                    hnorm_meter=hnorm_meter,
                )
            )
            s += f' | Bits/dim {bpd_meter.val:.4f}({bpd_meter.avg:.4f})'
            mprint(s)

        del x
        torch.cuda.empty_cache()
        gc.collect()


# noinspection PyUnusedLocal
def validate(epoch, model, data_loader, ema, device):
    """
    Evaluates the cross entropy between p_data and p_model.
    """
    bpd_meter = utils.AverageMeter()

    if ema is not None:
        ema.swap()

    model.eval()

    start = time.time()
    with torch.no_grad():
        for i, (x, y) in enumerate(tqdm(data_loader)):
            x = x.to(device)
            bpd = compute_loss(x, model)
            bpd_meter.update(bpd.item(), x.size(0))
    val_time = time.time() - start

    if ema is not None:
        ema.swap()

    return val_time, bpd_meter.avg


# noinspection PyShadowingNames
def main(rank, world_size, args):
    setup(rank, world_size, args.port)

    # setup logger
    if rank == 0:
        utils.makedirs(args.save)
        logger = utils.get_logger(os.path.join(args.save, "logs"))

    def mprint(msg):
        if rank == 0:
            logger.info(msg)

    mprint(args)

    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')

    if device.type == 'cuda':
        mprint('Found {} CUDA devices.'.format(torch.cuda.device_count()))
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            mprint('{} \t Memory: {:.2f}GB'.format(props.name, props.total_memory / (1024 ** 3)))
    else:
        mprint('WARNING: Using device {}'.format(device))

    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    if device.type == 'cuda':
        torch.cuda.manual_seed(args.seed + rank)

    mprint('Loading dataset {}'.format(args.data))
    # Dataset and hyperparameters
    if args.data == 'cifar10':
        im_dim = 3

        transform_train = transforms.Compose([
            transforms.Resize(args.imagesize),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            add_noise if args.add_noise else identity,
        ])
        transform_test = transforms.Compose([
            transforms.Resize(args.imagesize),
            transforms.ToTensor(),
            add_noise if args.add_noise else identity,
        ])

        init_layer = flows.LogitTransform(0.05)
        train_set = vdsets.SVHN(args.dataroot, download=True, split="train", transform=transform_train)
        sampler = torch.utils.data.distributed.DistributedSampler(train_set)
        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=args.batchsize,
            sampler=sampler,
        )
        test_loader = torch.utils.data.DataLoader(
            vdsets.SVHN(args.dataroot, download=True, split="test", transform=transform_test),
            batch_size=args.val_batchsize,
            shuffle=False,
        )

    elif args.data == 'mnist':
        im_dim = 1
        init_layer = flows.LogitTransform(1e-6)
        train_set = datasets.MNIST(
            args.dataroot, train=True, transform=transforms.Compose([
                transforms.Resize(args.imagesize),
                transforms.ToTensor(),
                add_noise if args.add_noise else identity,
            ])
        )
        sampler = torch.utils.data.distributed.DistributedSampler(train_set)
        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=args.batchsize,
            sampler=sampler,
        )
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(
                args.dataroot, train=False, transform=transforms.Compose([
                    transforms.Resize(args.imagesize),
                    transforms.ToTensor(),
                    add_noise if args.add_noise else identity,
                ])
            ),
            batch_size=args.val_batchsize,
            shuffle=False,
        )
    else:
        raise Exception(f'dataset not one of mnist / cifar10, got {args.data}')

    mprint('Dataset loaded.')
    mprint('Creating model.')

    input_size = (args.batchsize, im_dim, args.imagesize, args.imagesize)

    model = MultiscaleFlow(
        input_size,
        block_fn=partial(cpflow_block_fn, block_type=args.block_type, dimh=args.dimh,
                         num_hidden_layers=args.num_hidden_layers, icnn_version=args.icnn,
                         num_pooling=args.num_pooling),
        n_blocks=list(map(int, args.nblocks.split('-'))),
        factor_out=args.factor_out,
        init_layer=init_layer,
        actnorm=args.actnorm,
        fc_end=args.fc_end,
        glow=args.glow,
    )
    model.to(device)

    model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    ema = utils.ExponentialMovingAverage(model)

    mprint(model)
    mprint('EMA: {}'.format(ema))

    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=args.wd)

    # Saving and resuming
    best_test_bpd = math.inf
    begin_epoch = 0

    most_recent_path = os.path.join(args.save, 'models', 'most_recent.pth')
    checkpt_exists = os.path.exists(most_recent_path)
    if checkpt_exists:
        mprint(f"Resuming from {most_recent_path}")

        # deal with data-dependent initialization like actnorm.
        with torch.no_grad():
            x = torch.rand(8, *input_size[1:]).to(device)
            model(x)

        checkpt = torch.load(most_recent_path)
        begin_epoch = checkpt["epoch"] + 1

        model.module.load_state_dict(checkpt["state_dict"])
        ema.set(checkpt['ema'])
        optimizer.load_state_dict(checkpt["opt_state_dict"])
    elif args.resume:
        mprint(f"Resuming from {args.resume}")

        # deal with data-dependent initialization like actnorm.
        with torch.no_grad():
            x = torch.rand(8, *input_size[1:]).to(device)
            model(x)

        checkpt = torch.load(args.resume)
        begin_epoch = checkpt["epoch"] + 1

        model.module.load_state_dict(checkpt["state_dict"])
        ema.set(checkpt['ema'])
        optimizer.load_state_dict(checkpt["opt_state_dict"])

    mprint(optimizer)

    batch_time = utils.RunningAverageMeter(0.97)
    bpd_meter = utils.RunningAverageMeter(0.97)
    gnorm_meter = utils.RunningAverageMeter(0.97)
    cg_meter = utils.RunningAverageMeter(0.97)
    hnorm_meter = utils.RunningAverageMeter(0.97)

    update_lr(optimizer, 0, args)

    # for visualization
    fixed_x = next(iter(train_loader))[0][:8].to(device)
    fixed_z = torch.randn(8, im_dim * args.imagesize * args.imagesize).to(fixed_x)
    if rank == 0:
        utils.makedirs(os.path.join(args.save, 'figs'))
        # visualize(model, fixed_x, fixed_z, os.path.join(args.save, 'figs', 'init.png'))
    for epoch in range(begin_epoch, args.nepochs):
        sampler.set_epoch(epoch)
        flows.CG_ITERS_TRACER.clear()
        flows.HESS_NORM_TRACER.clear()
        mprint('Current LR {}'.format(optimizer.param_groups[0]['lr']))
        train(epoch, train_loader, model, optimizer, bpd_meter, gnorm_meter, cg_meter, hnorm_meter, batch_time, ema,
              device, mprint, world_size, args)
        val_time, test_bpd = validate(epoch, model, test_loader, ema, device)
        mprint('Epoch: [{0}]\tTime {1:.2f} | Test bits/dim {test_bpd:.4f}'.format(epoch, val_time, test_bpd=test_bpd))

        if rank == 0:
            utils.makedirs(os.path.join(args.save, 'figs'))
            visualize(model, fixed_x, fixed_z, os.path.join(args.save, 'figs', f'{epoch}.png'))

            utils.makedirs(os.path.join(args.save, "models"))
            if test_bpd < best_test_bpd:
                best_test_bpd = test_bpd
                torch.save({
                    'epoch': epoch,
                    'state_dict': model.module.state_dict(),
                    'opt_state_dict': optimizer.state_dict(),
                    'args': args,
                    'ema': ema,
                    'test_bpd': test_bpd,
                }, os.path.join(args.save, 'models', 'best_model.pth'))

        if rank == 0:
            torch.save({
                'epoch': epoch,
                'state_dict': model.module.state_dict(),
                'opt_state_dict': optimizer.state_dict(),
                'args': args,
                'ema': ema,
                'test_bpd': test_bpd,
            }, os.path.join(args.save, 'models', 'most_recent.pth'))

    cleanup()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='cifar10', choices=['mnist', 'cifar10'])
    parser.add_argument('--dataroot', type=str, default='data')
    parser.add_argument('--imagesize', type=int, default=None)
    parser.add_argument('--add_noise', type=eval, choices=[True, False], default=True)

    parser.add_argument('--block_type', choices=["coupling", "cvx"], default="cvx")
    parser.add_argument('--nblocks', type=str, default='3-3-3')
    parser.add_argument('--factor_out', type=eval, choices=[True, False], default=False)
    parser.add_argument('--actnorm', type=eval, choices=[True, False], default=True)
    parser.add_argument('--fc_end', type=eval, choices=[True, False], default=False)
    parser.add_argument('--glow', type=eval, choices=[True, False], default=False)

    parser.add_argument('--dimh', type=int, default=64)
    parser.add_argument('--num_hidden_layers', type=int, default=3)
    parser.add_argument('--num_pooling', type=int, default=0)
    parser.add_argument("--icnn", type=int, choices=[1, 2, 3], default=1)

    parser.add_argument('--nepochs', help='Number of epochs for training', type=int, default=100)
    parser.add_argument('--batchsize', help='Minibatch size', type=int, default=64)
    parser.add_argument('--lr', help='Learning rate', type=float, default=1e-3)
    parser.add_argument('--wd', help='Weight decay', type=float, default=1e-6)
    parser.add_argument('--warmup_iters', type=int, default=0)
    parser.add_argument('--save', help='directory to save results', type=str, default='test')
    parser.add_argument('--val_batchsize', help='minibatch size', type=int, default=200)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--ema_val', type=eval, choices=[True, False], default=True)
    parser.add_argument("--fast_adjoint", type=eval, choices=[True, False], default=False)

    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--ngpus', type=int, default=1)
    parser.add_argument('--port', type=int, default=None)
    parser.add_argument('--logfreq', help='Print progress every so iterations', type=int, default=20)
    parser.add_argument('--vis_freq', help='Visualize progress every so iterations', type=int, default=500)
    args = parser.parse_args()

    # torch.set_default_dtype(torch.float64)

    if args.port is None:
        args.port = int(np.random.randint(10000, 20000))

    if args.imagesize is None:
        if args.data == 'mnist':
            args.imagesize = 28
        elif args.data == 'cifar10':
            args.imagesize = 32

    # Random seed
    if args.seed is None:
        args.seed = np.random.randint(100000)

    # Top-level logger for logging exceptions into the log file.
    utils.makedirs(args.save)
    logger = utils.get_logger(os.path.join(args.save, "logs"))

    # noinspection PyBroadException
    try:
        mp.set_start_method("forkserver")
        mp.spawn(main,
                 args=(args.ngpus, args),
                 nprocs=args.ngpus,
                 join=True)
    except Exception:
        import traceback

        logger.error(traceback.format_exc())
        sys.exit(1)
