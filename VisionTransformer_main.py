import random  # random utilities for reproducibility/control
import tqdm  # progress bar utility for loops
import numpy as np  # numerical operations

import torch  # main PyTorch library
import torch.optim as optim  # optimizer module from PyTorch
from TrainValidateWrapper import TrainValidateWrapper  # wrapper around model for training/validation
from models.SimpleTransformer import SimpleTransformer  # transformer model definition
from PatchEmbedding import PatchEmbedding_CNN  # optional patch embedding implementation
import Utils  # project utilities (data loaders etc.)
import sys  # system utilities (exit)
import math  # math utilities
import os  # operating system utilities (paths, file ops)

#cuda diagnostic
# choose CUDA if available
device = torch.device("cuda")

# ------constants------------
NUM_EPOCHS = 20  # number of training epochs
BATCH_SIZE = 256  # batch size for training/validation
GRADIENT_ACCUMULATE_EVERY = 1  # accumulate gradients over this many mini-batches
LEARNING_RATE = 1e-4  # learning rate (note: author said 1e-3 doesn't learn)
VALIDATE_EVERY = 1  # run validation every N epochs
SEQ_LENGTH = 197  # sequence length (14*14 patches + 1 cls token)
RESUME_TRAINING = False  # whether to resume from checkpoint or start fresh
#---------------------------

best_test_accuracy = 0  # global to track best test accuracy seen



def count_parameters(model):  # count number of trainable parameters in the model
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def configure_optimizers(mymodel):
    """
    Separates model parameters into two groups: those that should receive
    weight decay (regularization) and those that shouldn't (biases, LayerNorm,
    embeddings). Returns an AdamW optimizer configured with these groups.
    """

    # separate out all parameters to those that will and won't experience regularizing weight decay
    decay = set()  # names of params that will be decayed
    no_decay = set()  # names of params that will NOT be decayed
    whitelist_weight_modules = (torch.nn.Linear,)  # module types whose weights should be decayed
    blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)  # module types whose weights should not be decayed

    for mn, m in mymodel.named_modules():  # iterate modules with their module-name
        for pn, p in m.named_parameters():  # iterate parameters under the module
            fpn = '%s.%s' % (mn, pn) if mn else pn  # full parameter name (module.param)
            # note: named_modules/named_parameters are recursive; we use fpn to know parent module
            if pn.endswith('bias'):  # biases should not be weight-decayed
                no_decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):  # weights of whitelist modules get decay
                decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):  # blacklist weights do not get decay
                no_decay.add(fpn)
            elif fpn.startswith('model.token_emb'):  # specific token embedding handling
                no_decay.add(fpn)  # ensure token embeddings are not decayed

    # validate that we considered every parameter
    param_dict = {pn: p for pn, p in mymodel.named_parameters()}  # map of all parameter names to tensors
    inter_params = decay & no_decay  # intersection should be empty
    union_params = decay | no_decay  # union should cover all params
    assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
    assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" % (str(param_dict.keys() - union_params),)

    # create the pytorch optimizer object with two parameter groups
    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": 0.1},  # decayed params
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},  # non-decayed params
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=LEARNING_RATE, betas=(0.9, 0.95))  # AdamW optimizer
    return optimizer


def main():
    global best_test_accuracy
    vision_model = SimpleTransformer(
        dim=768,  # embedding dimension
        num_unique_tokens=10,  # number of classes/tokens (CIFAR-10 -> 10)
        num_layers=12,  # transformer layers
        heads=8,  # attention heads
        max_seq_len=SEQ_LENGTH,  # maximum sequence length
    ).to(device)

    model = TrainValidateWrapper(vision_model)
    model.to(device)
    for n, p in model.named_parameters():
        if p.device.type != device.type:
            print('PARAM DEVICE MISMATCH:', n, p.device)
    print('Model built on device:', next(model.parameters()).device)

    pcount = count_parameters(model)  # count trainable params
    print("count of parameters in the model = ", pcount / 1e6, " million")  # print parameter count

    # get data loaders: train, validation, and test set
    train_loader, val_loader, testset = Utils.get_loaders_cifar(dataset_type="CIFAR10", img_width=224, img_height=224, batch_size=BATCH_SIZE)

    # create optimizer using the configure_optimizers helper
    optim = configure_optimizers(model)

    # --------training---------
    if RESUME_TRAINING == False:  # start from epoch 0 if not resuming
        start = 0
    else:
        checkpoint_data = torch.load('checkpoint/visiontrans_model.pt')  # load checkpoint
        model.load_state_dict(checkpoint_data['state_dict'])  # restore model weights
        optim.load_state_dict(checkpoint_data['optimizer'])  # restore optimizer state
        start = checkpoint_data['epoch']  # resume epoch
        best_test_accuracy = checkpoint_data['test_acc']  # restore best accuracy
        print('best test accuracy from restored model=', best_test_accuracy)

    for i in tqdm.tqdm(range(start, NUM_EPOCHS), mininterval=10., desc='training'):  # epoch loop with progress bar
        for k, data in enumerate(train_loader):  # iterate over training batches
            model.train()  # set model to training mode
            total_loss = 0  # reset loss accumulator for this iteration
            for __ in range(GRADIENT_ACCUMULATE_EVERY):  # gradient accumulation loop (usually 1)
                x, y = data  # get inputs and labels from batch
                x = x.to(device)
                y = y.to(device)
                loss = model(x, y)  # forward + loss computed by wrapper
                loss.backward()  # backpropagate loss
            if (k % 500 == 0):  # periodic logging
                print(f'training loss: {loss.item()} -- iterationh = {k}')

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # gradient clipping
            optim.step()  # optimizer step
            optim.zero_grad()  # zero gradients for next step

        if i % VALIDATE_EVERY == 0:  # validation checkpoint
            model.eval()  # set model to evaluation mode
            val_count = 3000  # (unused) example variable for validation subset size
            total_count = 0  # total number of validation samples processed
            count_correct = 0  # number of correct predictions
            with torch.no_grad():  # disable grad for validation
                for v, data in enumerate(val_loader):  # iterate validation batches
                    x, y = data  # get inputs and labels
                    x = x.to(device)
                    y = y.to(device)
                    count_correct = count_correct + model.validate(x, y)  # accumulate correct predictions
                    total_count = total_count + x.shape[0]  # accumulate total samples
                accuracy = (count_correct / total_count) * 100  # compute accuracy in percent
                print("\n-------------Test Accuracy = ", accuracy, "\n")  # print accuracy
            if accuracy > best_test_accuracy:  # if we've improved
                print("----------saving model-----------------")
                checkpoint_data = {
                    'epoch': i,  # current epoch
                    'state_dict': model.state_dict(),  # model weights
                    'optimizer': optim.state_dict(),  # optimizer state
                    'test_acc': accuracy  # achieved accuracy
                }
                ckpt_path = os.path.join("checkpoint/visiontrans_model.pt")  # checkpoint path
                torch.save(checkpoint_data, ckpt_path)  # save checkpoint
                best_test_accuracy = accuracy  # update best accuracy
            model.train()  # go back to training mode

    # end of training loop


if __name__ == "__main__":  # run main when executed as script
    sys.exit(int(main() or 0))