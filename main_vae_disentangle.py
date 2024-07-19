import argparse
import datetime
import os
import shutil
import time
import numpy as np
import dataset
import mlconfig
import torch
import util
import madrys
import models
from evaluator import Evaluator
from torchvision import datasets, transforms
import random
import math

random.seed(0)

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

class Trainer():
    def __init__(self, criterion, data_loader, logger, config, global_step=0,
                 target='train_dataset'):
        self.criterion = criterion
        self.data_loader = data_loader
        self.logger = logger
        self.config = config
        self.log_frequency = config.log_frequency if config.log_frequency is not None else 100
        self.loss_meters = util.AverageMeter()
        self.acc_meters = util.AverageMeter()
        self.acc5_meters = util.AverageMeter()
        self.global_step = global_step
        self.target = target
        print(self.target)

    def _reset_stats(self):
        self.loss_meters = util.AverageMeter()
        self.acc_meters = util.AverageMeter()
        self.acc5_meters = util.AverageMeter()

    def train(self, epoch, model, model_s, model_c, criterion, optimizer, random_noise=None):
        model.train()
        model_s.eval()
        if epoch == 0:
            idx = 0
            noise = np.zeros((50000, 3, 32, 32))
        for i, (images, labels) in enumerate(self.data_loader[self.target]):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            if random_noise is not None:
                random_noise = random_noise.detach().to(device)
                for i in range(len(labels)):
                    class_index = labels[i].item()
                    images[i] += random_noise[class_index].clone()
                    images[i] = torch.clamp(images[i], 0, 1)
            if epoch == 0:
                if model_s is not None:
                    logits, rec, mu, log_var, perturbation = model_s(images, labels)
                    images = (images - perturbation).clamp(0,1).detach()
                    if iters > 0:
                        logits, rec, mu, log_var, perturbation = model_s(images, labels)
                        images = rec.clamp(0,1).detach()
                    for k in range(images.shape[0]):
                        datasets_generator.datasets['train_dataset'].data[idx] = np.clip(255 * images[k].cpu().permute(1,2,0).numpy(), a_min=0, a_max=255).astype(np.uint8)
                        noise[idx] = perturbation[k].detach().cpu().numpy()
                        idx += 1
            else:
                if iters == 0:
                    break
                start = time.time()
                log_payload = self.train_batch(images, labels, model, optimizer)
                end = time.time()
                time_used = end - start
                if self.global_step % self.log_frequency == 0:
                    display = util.log_display(epoch=epoch,
                                               global_step=self.global_step,
                                               time_elapse=time_used,
                                               **log_payload)
                    self.logger.info(display)
                self.global_step += 1
        if epoch == 0:
            datasets_generator.datasets['train_dataset'].transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor()])
            if hasattr(datasets_generator.datasets['train_dataset'], 'array'):
                datasets_generator.datasets['train_dataset'].array = True
            self.data_loader = datasets_generator.getDataLoader()
            noise = torch.tensor(noise)
            torch.save(noise, os.path.join(args.exp_name, 'perturbation{}.pt'.format(iters)))
            del noise
            print (idx)
        return self.global_step

    def train_batch(self, images, labels, model, optimizer):
        model.zero_grad()
        optimizer.zero_grad()
        if isinstance(self.criterion, torch.nn.CrossEntropyLoss) or isinstance(self.criterion, models.CutMixCrossEntropyLoss):
            logits = model(images)
            loss = self.criterion(logits, labels)
        else:
            logits, loss = self.criterion(model, images, labels, optimizer)
        if isinstance(self.criterion, models.CutMixCrossEntropyLoss):
            _, labels = torch.max(labels.data, 1)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.grad_clip)
        optimizer.step()
        if logits.shape[1] >= 5:
            acc, acc5 = util.accuracy(logits, labels, topk=(1, 5))
            acc, acc5 = acc.item(), acc5.item()
        else:
            acc, = util.accuracy(logits, labels, topk=(1,))
            acc, acc5 = acc.item(), 1
        self.loss_meters.update(loss.item(), labels.shape[0])
        self.acc_meters.update(acc, labels.shape[0])
        self.acc5_meters.update(acc5, labels.shape[0])
        payload = {"acc": acc,
                   "acc_avg": self.acc_meters.avg,
                   "loss": loss,
                   "loss_avg": self.loss_meters.avg,
                   "lr": optimizer.param_groups[0]['lr'],
                   "|gn|": grad_norm}
        return payload

class VAE_Trainer():
    def __init__(self, data_loader, logger, config, global_step=0,
                 target='train_dataset'):
        self.data_loader = data_loader
        self.logger = logger
        self.config = config
        self.log_frequency = config.log_frequency if config.log_frequency is not None else 100
        self.loss_meters = util.AverageMeter()
        self.loss_rec_meters = util.AverageMeter()
        self.loss_rec_p_meters = util.AverageMeter()
        self.loss_p_meters = util.AverageMeter()
        self.loss_cls_meters = util.AverageMeter()
        self.loss_cls_p_meters = util.AverageMeter()
        self.loss_cls_p2_meters = util.AverageMeter()
        self.loss_cls_purified_meters = util.AverageMeter()
        self.kld_loss_meters = util.AverageMeter()
        self.psnr_meters = util.AverageMeter()
        self.psnr_p_meters = util.AverageMeter()
        self.global_step = global_step
        self.target = target
        print(self.target)

    def _reset_stats(self):
        self.loss_meters = util.AverageMeter()
        self.loss_rec_meters = util.AverageMeter()
        self.loss_rec_p_meters = util.AverageMeter()
        self.loss_p_meters = util.AverageMeter()
        self.loss_cls_meters = util.AverageMeter()
        self.loss_cls_p_meters = util.AverageMeter()
        self.loss_cls_p2_meters = util.AverageMeter()
        self.loss_cls_purified_meters = util.AverageMeter()
        self.kld_loss_meters = util.AverageMeter()
        self.psnr_meters = util.AverageMeter()
        self.psnr_p_meters = util.AverageMeter()

    def train(self, epoch, model, model_c, optimizer):
        model.train()
        model_c.eval()
        for i, (images, labels) in enumerate(self.data_loader[self.target]):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            start = time.time()
            log_payload = self.train_batch(images, labels, model, model_c, optimizer, epoch)
            end = time.time()
            time_used = end - start
            if self.global_step % self.log_frequency == 0:
                display = util.log_display(epoch=epoch,
                                           global_step=self.global_step,
                                           time_elapse=time_used,
                                           **log_payload)
                self.logger.info(display)
            self.global_step += 1
        return self.global_step

    def train_batch(self, images, labels, model, model_c, optimizer, epoch):
        if iters == 0:
            kld_target = 1.0
        else:
            kld_target = args.kd
        kl_loss = nn.KLDivLoss(reduction="batchmean")

        model.zero_grad()
        optimizer.zero_grad()
        logits, rec, mu, log_var, perturbation = model(images, labels)
        loss_rec = F.mse_loss(rec, images)
        loss_rec_p = F.mse_loss((rec + perturbation).clamp(0,1), images)
        loss_p = F.mse_loss(perturbation, torch.zeros_like(perturbation).cuda())
        
        loss_cls = F.cross_entropy(model_c(rec), labels)
        loss_cls_p = F.cross_entropy(model_c((rec + perturbation).clamp(0,1)), labels)
        loss_cls_p2 = F.cross_entropy(model_c((images - perturbation).clamp(0,1)), labels)

        loss_cls_purified = kl_loss(F.log_softmax(model_c((images - perturbation).clamp(0,1)), dim = 1), torch.ones_like(logits).float().cuda()/logits.shape[1])
        # kld_weight =  0.001
        if kld_target > 0.99:
            kld_weight = 0.01
        else:
            kld_weight = 0.5
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = (0,1,2))
        loss = loss_rec + loss_rec_p + kld_weight * max(kld_loss, kld_target)
        if epoch > 10:
            loss = loss + max(loss_p, 0.002)

        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.grad_clip)
        optimizer.step()

        psnr = -10 * math.log10(F.mse_loss(rec, images).item())
        psnr_p = -10 * math.log10(F.mse_loss((rec + perturbation).clamp(0,1), images).item())
        self.loss_meters.update(loss.item(), labels.shape[0])
        self.loss_rec_meters.update(loss_rec.item(), labels.shape[0])
        self.loss_rec_p_meters.update(loss_rec_p.item(), labels.shape[0])
        self.loss_p_meters.update(loss_p.item(), labels.shape[0])
        self.loss_cls_meters.update(loss_cls.item(), labels.shape[0])
        self.loss_cls_p_meters.update(loss_cls_p.item(), labels.shape[0])
        self.loss_cls_p2_meters.update(loss_cls_p2.item(), labels.shape[0])
        self.loss_cls_purified_meters.update(loss_cls_purified.item(), labels.shape[0])
        self.kld_loss_meters.update(kld_loss.item(), labels.shape[0])
        self.psnr_meters.update(psnr, labels.shape[0])
        self.psnr_p_meters.update(psnr_p, labels.shape[0])
        payload = {"psnr": psnr,
                   "psnr_avg": self.psnr_meters.avg,
                   "psnr_p": psnr_p,
                   "psnr_p_avg": self.psnr_p_meters.avg,
                   "loss": loss,
                   "loss_avg": self.loss_meters.avg,
                   "loss_rec": loss_rec,
                   "loss_rec_avg": self.loss_rec_meters.avg,
                   "loss_rec_p": loss_rec_p,
                   "loss_rec_p_avg": self.loss_rec_p_meters.avg,
                   "loss_p": loss_p,
                   "loss_p_avg": self.loss_p_meters.avg,
                   "loss_cls": loss_cls,
                   "loss_cls_avg": self.loss_cls_meters.avg,
                   "loss_cls_p": loss_cls_p,
                   "loss_cls_p_avg": self.loss_cls_p_meters.avg,
                   "loss_cls_p2": loss_cls_p2,
                   "loss_cls_p2_avg": self.loss_cls_p2_meters.avg,
                   "loss_cls_purified": loss_cls_purified,
                   "loss_cls_purified_avg": self.loss_cls_purified_meters.avg,
                   "kld_loss": kld_loss,
                   "kld_loss_avg": self.kld_loss_meters.avg,
                   "lr": optimizer.param_groups[0]['lr'],
                   "|gn|": grad_norm}
        return payload


mlconfig.register(madrys.MadrysLoss)
import torch.nn as nn
import torch.nn.functional as F

# General Options
parser = argparse.ArgumentParser(description='ClasswiseNoise')
parser.add_argument('--seed', type=int, default=0, help='seed')
parser.add_argument('--version', type=str, default="resnet18")
parser.add_argument('--version_s', type=str, default="resnet18_s")
parser.add_argument('--exp_name', type=str, default="test_exp")
parser.add_argument('--config_path', type=str, default='configs/cifar10')
parser.add_argument('--load_model', action='store_true', default=False)
parser.add_argument('--data_parallel', action='store_true', default=False)
parser.add_argument('--train', action='store_true', default=False)
parser.add_argument('--save_frequency', default=-1, type=int)
# Datasets Options
parser.add_argument('--train_face', action='store_true', default=False)
parser.add_argument('--train_portion', default=1.0, type=float)
parser.add_argument('--train_batch_size', default=128, type=int, help='perturb step size')
parser.add_argument('--eval_batch_size', default=256, type=int, help='perturb step size')
parser.add_argument('--num_of_workers', default=8, type=int, help='workers for loader')
parser.add_argument('--train_data_type', type=str, default='CIFAR10')
parser.add_argument('--test_data_type', type=str, default='CIFAR10')
parser.add_argument('--train_data_path', type=str, default='../datasets')
parser.add_argument('--test_data_path', type=str, default='../datasets')
parser.add_argument('--perturb_type', default='classwise', type=str, choices=['classwise', 'samplewise'], help='Perturb type')
parser.add_argument('--patch_location', default='center', type=str, choices=['center', 'random'], help='Location of the noise')
parser.add_argument('--poison_rate', default=1.0, type=float)
parser.add_argument('--perturb_tensor_filepath', default=None, type=str)
parser.add_argument('--identity', default='race', type=str, choices=['race', 'age', 'gender', 'class', 'bias'])
parser.add_argument('--kd', default=2.5, type=float, help='target kld loss')
parser.add_argument('--sample_wise_rp', action='store_false')
parser.add_argument('--use_y', action='store_false')
parser.add_argument('--spatial_emb', action='store_true')
global args
args = parser.parse_args()

# Set up Experiments
if args.exp_name == '':
    args.exp_name = 'exp_' + datetime.datetime.now()

exp_path = os.path.join(args.exp_name, args.version)
log_file_path = os.path.join(exp_path, args.version)
checkpoint_path = os.path.join(exp_path, 'checkpoints')
checkpoint_path_file = os.path.join(checkpoint_path, args.version)
util.build_dirs(exp_path)
util.build_dirs(checkpoint_path)
logger = util.setup_logger(name=args.version, log_file=log_file_path + ".log")

# CUDA Options
logger.info("PyTorch Version: %s" % (torch.__version__))
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda')
    device_list = [torch.cuda.get_device_name(i) for i in range(0, torch.cuda.device_count())]
    logger.info("GPU List: %s" % (device_list))
else:
    device = torch.device('cpu')

# Load Exp Configs
config_file = os.path.join(args.config_path, args.version)+'.yaml'
config = mlconfig.load(config_file)
config.set_immutable()
for key in config:
    logger.info("%s: %s" % (key, config[key]))
shutil.copyfile(config_file, os.path.join(exp_path, args.version+'.yaml'))


def train(starting_epoch, model, model_s, model_c, optimizer, scheduler, criterion, trainer, evaluator, ENV, data_loader):
    for epoch in range(starting_epoch, config.epochs + 1):
        logger.info("")
        logger.info("="*20 + "Training Epoch %d" % (epoch) + "="*20)

        # Train
        ENV['global_step'] = trainer.train(epoch, model, model_s, model_c, criterion, optimizer)
        ENV['train_history'].append(trainer.acc_meters.avg*100)
        if epoch > 0:
            scheduler.step()

        # Eval
        logger.info("="*20 + "Eval Epoch %d" % (epoch) + "="*20)
        is_best = False
        if not args.train_face:
            evaluator.eval(epoch, model)
            payload = ('Eval Loss:%.4f\tEval acc: %.2f' % (evaluator.loss_meters.avg, evaluator.acc_meters.avg*100))
            logger.info(payload)
            ENV['eval_history'].append(evaluator.acc_meters.avg*100)
            ENV['curren_acc'] = evaluator.acc_meters.avg*100
            ENV['cm_history'].append(evaluator.confusion_matrix.cpu().numpy().tolist())
            # Reset Stats
            trainer._reset_stats()
            evaluator._reset_stats()
        else:
            pass

        # Save Model
        target_model = model.module if args.data_parallel else model
        util.save_model(ENV=ENV,
                        epoch=epoch,
                        model=target_model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        is_best=is_best,
                        filename=checkpoint_path_file)
        logger.info('Model Saved at %s', checkpoint_path_file)

        if args.save_frequency > 0 and epoch % args.save_frequency == 0:
            filename = checkpoint_path_file + '_epoch%d' % (epoch)
            util.save_model(ENV=ENV,
                            epoch=epoch,
                            model=target_model,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            filename=filename)
            logger.info('Model Saved at %s', filename)

    return


def main():
    model = config.model().to(device)
    # print (model)
    global datasets_generator
    datasets_generator = config.dataset(train_data_type=args.train_data_type,
                                        train_data_path=args.train_data_path,
                                        test_data_type=args.test_data_type,
                                        test_data_path=args.test_data_path,
                                        train_batch_size=args.train_batch_size,
                                        eval_batch_size=args.eval_batch_size,
                                        num_of_workers=args.num_of_workers,
                                        poison_rate=args.poison_rate,
                                        perturb_type=args.perturb_type,
                                        patch_location=args.patch_location,
                                        perturb_tensor_filepath=args.perturb_tensor_filepath,
                                        seed=args.seed, identity = args.identity)
    logger.info('Training Dataset: %s' % str(datasets_generator.datasets['train_dataset']))
    logger.info('Test Dataset: %s' % str(datasets_generator.datasets['test_dataset']))
    if 'Poison' in args.train_data_type:
        with open(os.path.join(exp_path, 'poison_targets.npy'), 'wb') as f:
            if not (isinstance(datasets_generator.datasets['train_dataset'], dataset.MixUp) or isinstance(datasets_generator.datasets['train_dataset'], dataset.CutMix)):
                poison_targets = np.array(datasets_generator.datasets['train_dataset'].poison_samples_idx)
                np.save(f, poison_targets)
                logger.info(poison_targets)
                logger.info('Poisoned: %d/%d' % (len(poison_targets), len(datasets_generator.datasets['train_dataset'])))
                logger.info('Poisoned samples idx saved at %s' % (os.path.join(exp_path, 'poison_targets')))
                logger.info('Poisoned Class %s' % (str(datasets_generator.datasets['train_dataset'].poison_class)))

    config_s_file = os.path.join(args.config_path, args.version_s)+'.yaml'
    config_s = mlconfig.load(config_s_file)
    config_s.set_immutable()

    global iters
    for it in range(2):
        iters = it
        datasets_generator.datasets['train_dataset'].transform = transforms.ToTensor()
        if args.train_portion == 1.0:
            data_loader = datasets_generator.getDataLoader(train_shuffle=True, train_drop_last=True)
            train_target = 'train_dataset'
        else:
            train_target = 'train_subset'
            data_loader = datasets_generator._split_validation_set(args.train_portion,
                                                                   train_shuffle=True,
                                                                   train_drop_last=True)
        
        model_s = config_s.model()
        model_s.sample_wise = args.sample_wise_rp
        model_s.use_y = args.use_y
        model_s.spatial_emb = args.spatial_emb
        model_s.set_spatial_emb()
        model_s = model_s.to(device)
        model_c = config.model().to(device)
        if args.data_parallel:
            model_s = torch.nn.DataParallel(model_s)
            model_c = torch.nn.DataParallel(model_c)

        if it == 0:
            filename = './{}/resnet18/checkpoints/resnet18'.format("_".join(args.exp_name.split("_")[:4]))
        else:
            filename = './{}/resnet18/checkpoints/resnet18'.format("_".join(args.exp_name.split("_")[:4]))
        checkpoint = util.load_model(filename=filename,
                                     model=model_c,
                                     optimizer=None,
                                     alpha_optimizer=None,
                                     scheduler=None)
        model_c.cam_register(layer = 'layer2')

        trainer = VAE_Trainer(data_loader, logger, config, target=train_target)
        optimizer = config_s.optimizer(model_s.parameters())
        scheduler = config_s.scheduler(optimizer)
        for epoch in range(0, config_s.epochs):
            logger.info("")
            logger.info("="*20 + "Training AE Epoch %d" % (epoch) + "="*20)

            # Train
            trainer.train(epoch, model_s, model_c, optimizer)
            scheduler.step()
            trainer._reset_stats()

        # Save Model_s
        target_model = model_s.module if args.data_parallel else model_s
        state = {'model_state_dict': target_model.state_dict()}
        torch.save(state, os.path.join(args.exp_name, 'model_s{}.pth'.format(iters)))

        datasets_generator.datasets['train_dataset'].transform = transforms.ToTensor()
        if args.train_portion == 1.0:
            data_loader = datasets_generator.getDataLoader(train_shuffle=False, train_drop_last=False)
            train_target = 'train_dataset'
        else:
            train_target = 'train_subset'
            data_loader = datasets_generator._split_validation_set(args.train_portion,
                                                                   train_shuffle=True,
                                                                   train_drop_last=True)

        logger.info("param size = %fMB", util.count_parameters_in_MB(model))
        optimizer = config.optimizer(model.parameters())
        scheduler = config.scheduler(optimizer)
        criterion = config.criterion()
        trainer = Trainer(criterion, data_loader, logger, config, target=train_target)
        evaluator = Evaluator(data_loader, logger, config)

        starting_epoch = 0
        ENV = {'global_step': 0,
               'best_acc': 0.0,
               'curren_acc': 0.0,
               'best_pgd_acc': 0.0,
               'train_history': [],
               'eval_history': [],
               'pgd_eval_history': [],
               'genotype_list': [],
               'cm_history': []}

        if args.load_model:
            checkpoint = util.load_model(filename=checkpoint_path_file,
                                         model=model,
                                         optimizer=optimizer,
                                         alpha_optimizer=None,
                                         scheduler=scheduler)
            starting_epoch = checkpoint['epoch']
            ENV = checkpoint['ENV']
            trainer.global_step = ENV['global_step']
            logger.info("File %s loaded!" % (checkpoint_path_file))

        if args.data_parallel:
            model = torch.nn.DataParallel(model)

        if args.train:
            train(starting_epoch, model, model_s, model_c, optimizer, scheduler, criterion, trainer, evaluator, ENV, data_loader)


if __name__ == '__main__':
    for arg in vars(args):
        logger.info("%s: %s" % (arg, getattr(args, arg)))
    start = time.time()
    main()
    end = time.time()
    cost = (end - start) / 86400
    payload = "Running Cost %.2f Days \n" % cost
    logger.info(payload)
