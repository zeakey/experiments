import matplotlib
matplotlib.use('Agg')

import argparse, time, logging

import numpy as np
import mxnet as mx

from mxnet import gluon, nd
from mxnet import autograd as ag
from mxnet.gluon import nn
from mxnet.gluon.data.vision import transforms

from gluoncv.model_zoo import get_model
from gluoncv.utils import makedirs, TrainingHistory
from gluoncv.data import transforms as gcv_transforms
from tensorboardX import SummaryWriter

# CLI
def parse_args():
    parser = argparse.ArgumentParser(description='Train a model for image classification.')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='training batch size per device (CPU/GPU).')
    parser.add_argument('--num-gpus', type=int, default=1,
                        help='number of gpus to use.')
    parser.add_argument('--model', type=str, default='cifar_resnet20_v2',
                        help='model to use. options are resnet and wrn. default is resnet.')
    parser.add_argument('-j', '--num-data-workers', dest='num_workers', default=4, type=int,
                        help='number of preprocessing workers')
    parser.add_argument('--num-epochs', type=int, default=200,
                        help='number of training epochs.')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='learning rate. default is 0.1.')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum value for optimizer, default is 0.9.')
    parser.add_argument('--wd', type=float, default=0.0001,
                        help='weight decay rate. default is 0.0001.')
    parser.add_argument('--lr-decay', type=float, default=0.1,
                        help='decay rate of learning rate. default is 0.1.')
    parser.add_argument('--lr-decay-period', type=int, default=0,
                        help='period in epoch for learning rate decays. default is 0 (has no effect).')
    parser.add_argument('--lr-decay-epoch', type=str, default='100,150',
                        help='epochs at which learning rate decays. default is 40,60.')
    parser.add_argument('--drop-rate', type=float, default=0.0,
                        help='dropout rate for wide resnet. default is 0.')
    parser.add_argument('--mode', type=str,
                        help='mode in which to train the model. options are imperative, hybrid')
    parser.add_argument('--save-period', type=int, default=10,
                        help='period in epoch of model saving.')
    parser.add_argument('--save-dir', type=str, default='tmp/tmp-0',
                        help='directory of saved models')
    parser.add_argument('--resume-from', type=str,
                        help='resume training from the model')
    parser.add_argument('--save-plot-dir', type=str, default='.',
                        help='the path to save the history plot')
    #
    parser.add_argument('--pruning-rate', type=float, default=0.1,
                        help='learning rate. default is 0.1.')
    opt = parser.parse_args()
    return opt

class Mask(object):
    def __init__(self, parameters, ratio):
        """
        parameters: dict containing parameters
        ratio: pruning ratio
        """
        self.mask = {}
        self.params = parameters
        self.ratio = ratio

        for name, p in self.params.items():
            self.mask[name] = mx.nd.ones_like(p.data())
    
    def update_mask(self):
        for name, p in self.params.items():
            pdata = p.data().reshape(p.data().shape[0], -1)
            num_pruned = int(pdata.shape[0] * self.ratio)
            norm = mx.nd.norm(pdata, ord=2, axis=1)
            indices_to_be_pruned = mx.nd.argsort(norm)[:num_pruned]
            self.mask[name] = mx.nd.ones_like(self.mask[name])
            self.mask[name][indices_to_be_pruned] = 0

    def forward_mask(self):
        for name, p in self.params.items():
            pdata = self.params[name].data()
            pdata = pdata * self.mask[name]
            p.set_data(pdata)
    
    def backward_mask(self):
        pass

def do_mask(param, mask):
    pass

def get_mask(weight, ratio=0.6):
    N = weight.shape[0]
    num_pruned = int(N * ratio)

    mask = np.ones_like(weight)
    weight = weight.reshape(weight.shape[0], -1)
    norm = np.sqrt((weight * weight).sum(axis=1))
    indices_to_be_pruned = np.argsort(norm)[:num_pruned].tolist()
    mask[indices_to_be_pruned, :, :, :] = 0

    return mask


def main():
    opt = parse_args()
    
    batch_size = opt.batch_size
    classes = 10

    num_gpus = opt.num_gpus
    batch_size *= max(1, num_gpus)
    context = [mx.gpu(i) for i in range(num_gpus)] if num_gpus > 0 else [mx.cpu()]
    num_workers = opt.num_workers

    lr_decay = opt.lr_decay
    lr_decay_epoch = [int(i) for i in opt.lr_decay_epoch.split(',')] + [np.inf]

    model_name = opt.model
    if model_name.startswith('cifar_wideresnet'):
        kwargs = {'classes': classes,
                'drop_rate': opt.drop_rate}
    else:
        kwargs = {'classes': classes}
    net = get_model(model_name, **kwargs)

    # for data and gradient mask
    conv_params = net.collect_params(".*conv*")

    if opt.resume_from:
        net.load_parameters(opt.resume_from, ctx = context)
    optimizer = 'nag'

    save_period = opt.save_period
    if opt.save_dir and save_period:
        save_dir = opt.save_dir
        makedirs(save_dir)
    else:
        save_dir = ''
        save_period = 0
    
    tfboard_writer = SummaryWriter(save_dir)

    plot_path = opt.save_plot_dir

    logging.basicConfig(level=logging.INFO)
    logging.info(opt)

    transform_train = transforms.Compose([
        gcv_transforms.RandomCrop(32, pad=4),
        transforms.RandomFlipLeftRight(),
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])

    def test(ctx, val_data):
        metric = mx.metric.Accuracy()
        for i, batch in enumerate(val_data):
            data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
            label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
            outputs = [net(X) for X in data]
            metric.update(label, outputs)
        return metric.get()

    def train(epochs, ctx):
        if isinstance(ctx, mx.Context):
            ctx = [ctx]
        net.initialize(mx.init.Xavier(), ctx=ctx)

        # dummy forward to initiate parameters
        dummy_data = mx.nd.zeros((10, 3, 224, 224), ctx=ctx[0])
        _ = net(dummy_data)
        # mask for pruning
        mask = Mask(conv_params, ratio=opt.pruning_rate)

        train_data = gluon.data.DataLoader(
            gluon.data.vision.CIFAR10(train=True).transform_first(transform_train),
            batch_size=batch_size, shuffle=True, last_batch='discard', num_workers=num_workers)

        val_data = gluon.data.DataLoader(
            gluon.data.vision.CIFAR10(train=False).transform_first(transform_test),
            batch_size=batch_size, shuffle=False, num_workers=num_workers)

        trainer = gluon.Trainer(net.collect_params(), optimizer,
                                {'learning_rate': opt.lr, 'wd': opt.wd, 'momentum': opt.momentum})
        metric = mx.metric.Accuracy()
        train_metric = mx.metric.Accuracy()
        loss_fn = gluon.loss.SoftmaxCrossEntropyLoss()
        train_history = TrainingHistory(['training-error', 'validation-error'])

        iteration = 0
        lr_decay_count = 0

        best_val_score = 0

        for epoch in range(epochs):
            tic = time.time()
            train_metric.reset()
            metric.reset()
            train_loss = 0
            num_batch = len(train_data)
            alpha = 1

            if epoch == lr_decay_epoch[lr_decay_count]:
                trainer.set_learning_rate(trainer.learning_rate*lr_decay)
                lr_decay_count += 1

            for i, batch in enumerate(train_data):
                data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
                label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)

                with ag.record():
                    output = [net(X) for X in data]
                    loss = [loss_fn(yhat, y) for yhat, y in zip(output, label)]
                for l in loss:
                    l.backward()
                trainer.step(batch_size)
                train_loss += sum([l.sum().asscalar() for l in loss])

                train_metric.update(label, output)
                name, acc = train_metric.get()
                iteration += 1

            mask.update_mask()
            mask.forward_mask()

            train_loss /= batch_size * num_batch
            name, acc = train_metric.get()
            name, val_acc = test(ctx, val_data)
            train_history.update([1-acc, 1-val_acc])
            train_history.plot(save_path='%s/%s_history.png'%(plot_path, model_name))

            if val_acc > best_val_score:
                best_val_score = val_acc
                net.save_parameters('%s/%.4f-cifar-%s-%d-best.params'%(save_dir, best_val_score, model_name, epoch))

            name, val_acc = test(ctx, val_data)
            logging.info('[Epoch %d] train=%f val=%f loss=%f time: %f' %
                (epoch, acc, val_acc, train_loss, time.time()-tic))
            
            # for k, v in conv_params.items():
            #     np_data = v.data().asnumpy()
            #     np_grad = v.grad().asnumpy()
            #     logging.info("Param name: %s, data-mean: %.5f, data-std: %.5f, grad-mean: %.5f, grad-std: %.5f" % (
            #         k, np_data.mean(), np_data.std(), np_grad.mean(), np_grad.std()
            #     ))

            tfboard_writer.add_scalar('train/acc', acc, epoch)
            tfboard_writer.add_scalar('train/loss', train_loss, epoch)
            tfboard_writer.add_scalar('val/acc', val_acc, epoch)

            if save_period and save_dir and (epoch + 1) % save_period == 0:
                net.save_parameters('%s/cifar10-%s-%d.params'%(save_dir, model_name, epoch))

        if save_period and save_dir:
            net.save_parameters('%s/cifar10-%s-%d.params'%(save_dir, model_name, epochs-1))



    if opt.mode == 'hybrid':
        net.hybridize()
    train(opt.num_epochs, context)

if __name__ == '__main__':
    main()
