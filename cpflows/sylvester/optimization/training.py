from __future__ import print_function
import torch
import gc

from torch.autograd import Variable
from cpflows.sylvester.optimization.loss import calculate_loss
from cpflows.sylvester.utils.visual_evaluation import plot_reconstructions
from cpflows.sylvester.utils.log_likelihood import calculate_likelihood

import numpy as np


def train(epoch, train_loader, model, opt, args):

    model.train()
    train_loss = np.zeros(len(train_loader))
    train_bpd = np.zeros(len(train_loader))

    num_data = 0

    # set warmup coefficient
    # beta = min([max((epoch * 1.) / max([args.warmup, 1.]), args.min_beta), args.max_beta])
    beta = min([(epoch * 1.) / max([args.warmup, 1.]), args.max_beta])
    print('beta = {:5.4f}'.format(beta))
    for batch_idx, (data, _) in enumerate(train_loader):

        if args.cuda:
            data = data.cuda()

        if args.dynamic_binarization:
            data = torch.bernoulli(data)
        data = data.double()

        data = Variable(data)
        data = data.view(-1, *args.input_size)

        opt.zero_grad()
        x_mean, z_mu, z_var, ldj, z0, zk = model(data)

        loss, rec, kl, bpd = calculate_loss(x_mean, data, z_mu, z_var, z0, zk, ldj, args, beta=beta)

        loss.backward()
        train_loss[batch_idx] = loss.item()
        train_bpd[batch_idx] = bpd

        opt.step()

        rec = rec.item()
        kl = kl.item()

        num_data += len(data)

        if batch_idx % args.log_interval == 0:
            if args.input_type == 'binary':
                print('Epoch: {:3d} [{:5d}/{:5d} ({:2.0f}%)]  \tLoss: {:11.6f}\trec: {:11.6f}\tkl: {:11.6f}'.format(
                    epoch, num_data, len(train_loader.sampler), 100. * batch_idx / len(train_loader),
                    loss.item(), rec, kl))
            else:
                perc = 100. * batch_idx / len(train_loader)
                tmp = 'Epoch: {:3d} [{:5d}/{:5d} ({:2.0f}%)] \tLoss: {:11.6f}\tbpd: {:8.6f}'
                print(tmp.format(epoch, num_data, len(train_loader.sampler), perc, loss.item(), bpd),
                      '\trec: {:11.3f}\tkl: {:11.6f}'.format(rec, kl))

        del data
        torch.cuda.empty_cache()
        gc.collect()

    if args.input_type == 'binary':
        print('====> Epoch: {:3d} Average train loss: {:.4f}'.format(
            epoch, train_loss.sum() / len(train_loader)))
    else:
        print('====> Epoch: {:3d} Average train loss: {:.4f}, average bpd: {:.4f}'.format(
            epoch, train_loss.sum() / len(train_loader), train_bpd.sum() / len(train_loader)))

    return train_loss


def evaluate(data_loader, model, args, testing=False, file=None, epoch=0):
    model.eval()
    loss = 0.
    batch_idx = 0
    bpd = 0.

    if args.input_type == 'binary':
        loss_type = 'elbo'
    else:
        loss_type = 'bpd'

    for data, _ in data_loader:
        batch_idx += 1

        if args.cuda:
            data = data.cuda()
        data = data.double()

        data = Variable(data, volatile=True)
        data = data.view(-1, *args.input_size)

        x_mean, z_mu, z_var, ldj, z0, zk = model(data)

        batch_loss, rec, kl, batch_bpd = calculate_loss(x_mean, data, z_mu, z_var, z0, zk, ldj, args)

        bpd += batch_bpd
        loss += batch_loss.item()

        # PRINT RECONSTRUCTIONS
        if batch_idx == 1 and testing is False:
            plot_reconstructions(data, x_mean, batch_loss, loss_type, epoch, args)

    loss /= len(data_loader)
    bpd /= len(data_loader)

    # Compute log-likelihood
    if testing:
        if args.flow == 'cpflow':
            # setting logdet estimation to brute force
            for f in model.flow.flows:
                f.no_bruteforce = False
        model.eval()

        log_likelihood, nll_bpd = 0.0, 0.0
        for data, _ in data_loader:
            batch_idx += 1

            if args.cuda:
                data = data.cuda()
            data = data.double()

            data = Variable(data, volatile=True)
            data = data.view(-1, *args.input_size)

            x_mean, z_mu, z_var, ldj, z0, zk = model(data)

            batch_loss, rec, kl, batch_bpd = calculate_loss(x_mean, data, z_mu, z_var, z0, zk, ldj, args)

            nll_bpd += batch_bpd
            log_likelihood += batch_loss.item()
        
        nll_bpd /= len(data_loader)
        log_likelihood /= len(data_loader)

    else:
        log_likelihood = None
        nll_bpd = None



    # Logging
    if args.input_type in ['multinomial']:
        bpd = loss / (np.prod(args.input_size) * np.log(2.))

    if file is None:
        if testing:
            print('====> Test set loss: {:.4f}'.format(loss))
            print('====> Test set log-likelihood: {:.4f}'.format(log_likelihood))

            if args.input_type != 'binary':
                print('====> Test set bpd (elbo): {:.4f}'.format(bpd))
                print('====> Test set bpd (log-likelihood): {:.4f}'.format(log_likelihood /
                                                                           (np.prod(args.input_size) * np.log(2.))))

        else:
            print('====> Validation set loss: {:.4f}'.format(loss))
            if args.input_type in ['multinomial']:
                print('====> Validation set bpd: {:.4f}'.format(bpd))
    else:
        with open(file, 'a') as ff:
            if testing:
                print('====> Test set loss: {:.4f}'.format(loss), file=ff)
                print('====> Test set log-likelihood: {:.4f}'.format(log_likelihood), file=ff)

                if args.input_type != 'binary':
                    print('====> Test set bpd: {:.4f}'.format(bpd), file=ff)
                    print('====> Test set bpd (log-likelihood): {:.4f}'.format(log_likelihood /
                                                                               (np.prod(args.input_size) * np.log(2.))),
                          file=ff)

            else:
                print('====> Validation set loss: {:.4f}'.format(loss), file=ff)
                if args.input_type != 'binary':
                    print('====> Validation set bpd: {:.4f}'.format(loss / (np.prod(args.input_size) * np.log(2.))),
                          file=ff)

    if not testing:
        return loss, bpd
    else:
        return log_likelihood, nll_bpd
