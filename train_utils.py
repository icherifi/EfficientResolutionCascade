import torch
from scores import AverageMeter
from scores import ProgressMeter, accuracy, Summary
import time


    
def train(data, model, criterion, optimizer, epoch):
    
    device = "cuda"
    data_time = AverageMeter('Time', ':6.3f', summary_type=Summary.SUM)
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    train_loader = data.data_batch["train"]
    
    progress = ProgressMeter(
        len(train_loader),
        [data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))
    
    # switch to train mode
    model.train()
    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        
        data_time.update(time.time() - end)
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do opt step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        end = time.time()
        
        
        if i % 64 == 0:
            progress.display(i + 1)
            
    progress.display_summary()