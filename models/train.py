import shutil
import os
import time
import torch
import torchvision.transforms as tf
import models.testset as ts

print_freq = 100

trf_train_norm = tf.Compose([tf.RandomCrop(32, padding=4), tf.RandomHorizontalFlip(), ts.normalize])
trf_norm = tf.Compose([ts.normalize])
tfasis = tf.Compose([])

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

        
def save_checkpoint(state, is_best, fdir):
    filepath = os.path.join(fdir, 'checkpoint.pth')
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(fdir, 'model_best.pth.tar'))


def adjust_learning_rate(optimizer, epoch):
    """For resnet, the lr starts from 0.1, and is divided by 10 at 80 and 120 epochs"""
    adjust_list = [150, 225]
    if epoch in adjust_list:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1  



def train(trainloader, model, criterion, optimizer, epoch):
    losses = AverageMeter()
    top1 = AverageMeter()

    model.train()

    end = time.time()
    for i, (input, target) in enumerate(trainloader):
        # measure data loading time

        input, target = input.cuda(), target.cuda()
        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec = accuracy(output, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time


    print(f'Epoch {epoch}, Time {time.time() - end:.3f}, Loss {losses.avg:.4f}, Prec {top1.avg:.3f}%')
    return {'loss_val': losses.val, 'loss_avg': losses.avg, 'prec_val': top1.val, 'prec_avg': top1.avg}

            

def validate(val_loader, model, criterion ):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
     
        input, target = input.cuda(), target.cuda()

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec = accuracy(output, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    print(f'Time {time.time()-end}, Loss {losses.avg:.4f}, Prec {top1.avg:.3f}%')
    return top1.avg



def eval_core(model, x, target, criterion, accf):
    o = model(x)
    loss = criterion(o, target)
    prec = accf(o, target)
    return loss, prec, o

def optstep(opt, loss):
    opt.zero_grad()
    loss.backward()
    opt.step()


def validate_training(val_loader, model, criterion, getopt,
        n_epoch = 1, lth=0, pth=100, nhist=1, 
        lradjust=adjust_learning_rate):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    accf = lambda x, y: accuracy(x, y)[0]

    end = time.time()
    hl = []
    ret = []
    for i, (input, target) in enumerate(val_loader):
        input, target = input.cuda(), target.cuda()
        prec = None
        epoch = 0
        optimizer = getopt(model, i)
        lradjust(optimizer, epoch)

        model.eval()
        loss, prec, _ = eval_core(model, input, target, criterion, accf)
        losses.update(loss.item(), input.size(0))
        top1.update(prec.item(), input.size(0))
        ret.append((prec.item(), input.size(0)))
        print(f'Initial: [{i}/{len(val_loader)}]\tLoss {loss.item()}\tPrec {prec.item()}')

        model.train()

        if len(hl) == nhist:
            hl.pop(0)

        hl.append((input, target))

        for epoch in range(0, n_epoch):
            ls = AverageMeter()
            tops = AverageMeter()
            for (ic, tc) in hl:
                lradjust(optimizer, epoch)
                loss, prec, _ = eval_core(model, ic, tc, criterion, accf)
                ls.update(loss.item(), ic.size(0))
                tops.update(prec.item(), ic.size(0))

                optstep(optimizer, loss)

                if epoch % print_freq == 0:  # This line shows how frequently print out the status. e.g., i%5 => every 5 batch, prints out
                    print(f'Test: [{i}/{len(val_loader)}]\tEpoch: {epoch}\tLoss {ls.avg}\tPrec {tops.avg}')

            if ls.avg <= lth and tops.avg >= pth or epoch == n_epoch - 1:
                batch_time.update(time.time() - end)
                end = time.time()

                print('Test: [{0}/{1}]\tEpoch: {epoch}\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Prec {top1.val:.3f}% ({top1.avg:.3f}%)'.format(
                    i, len(val_loader), epoch=epoch, batch_time=batch_time, loss=ls,
                    top1=tops))
                break


    print(' * Prec {top1.avg:.3f}%'.format(top1=top1))
    return top1.avg, ret



def validate_batch(val_loader, model, criterion ):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        input, target = input.cuda(), target.cuda()

        # compute output
        model.train()
        output = model(input)
        loss = criterion(output, target)
        model.eval()

        # measure accuracy and record loss
        prec = accuracy(output, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    print(f'Time {time.time()-end}, Loss {losses.avg:.4f}, Prec {top1.avg:.3f}%')
    return top1.avg


def validate_training_sep(val_loader, model, criterion, getopt,
        n_epoch=1, nhist=1, lradjust=adjust_learning_rate,
        trf_eval=tfasis, trf_train=tfasis):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    accf = lambda x, y: accuracy(x, y)[0]

    end = time.time()
    hl = []
    ret = []

    for i, (input, target) in enumerate(val_loader):
        hl.append((input, target))
        input, target = trf_eval(input.cuda()), target.cuda()
        prec = None
        epoch = 0

        model.eval()
        loss, prec, _ = eval_core(model, input, target, criterion, accf)
        losses.update(loss.item(), input.size(0))
        top1.update(prec.item(), input.size(0))
        ret.append((prec.item(), loss.item(), input.size(0)))

        print(f'Initial: [{i}/{len(val_loader)}]\tLoss {loss.item()}\tPrec {prec.item()}')

        if len(hl) < nhist and (i + 1) < len(val_loader):
            continue
        print(len(hl))

        optimizer = getopt(model, i)
        print(optimizer)
        model.train()
        for epoch in range(0, n_epoch):
            ls = AverageMeter()
            tops = AverageMeter()
            lradjust(optimizer, epoch)

            for j, (ic, tc) in enumerate(hl):
                ic, tc = ic.cuda(), tc.cuda()
                ic = trf_train(ic)

                output = model(ic)
                loss = criterion(output, tc)
                prec = accf(output, tc)
                ls.update(loss.item(), ic.size(0))
                tops.update(prec.item(), ic.size(0))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


            print(f'Time {time.time() - end}\tEpoch: {epoch}\tLoss {ls.avg:.4f}\tPrec {tops.avg:.3f}')


            if epoch == n_epoch - 1: #ls.avg <= lth and tops.avg >= pth or epoch == n_epoch - 1:
                batch_time.update(time.time() - end)
                end = time.time()

                print('Test: [{0}/{1}]\tEpoch: {epoch}\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Prec {top1.val:.3f}% ({top1.avg:.3f}%)'.format(
                    i, len(val_loader), epoch=epoch, batch_time=batch_time, loss=ls,
                    top1=tops))
                #break
        hl = []


    print(' * Prec {top1.avg:.3f}%'.format(top1=top1))
    return top1.avg, ret

