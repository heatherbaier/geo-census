import os
import time
import math
import shutil
import pickle

import torch
import torch.nn.functional as F

from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tensorboard_logger import configure, log_value

from model import RecurrentAttention
from utils import AverageMeter

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = "6"


def calc_reward(true, pred):

    quants = [true * .10, 
              true * .20, 
              true * .30, 
              true * .40, 
              true * .50, 
              true * .60, 
              true * .70, 
              true * .80, 
              true * .90]
    
    if true == 0:
        quants = [true + 1, 
                  true + 2, 
                  true + 3,
                  true + 4, 
                  true + 5, 
                  true + 6,
                  true + 7, 
                  true + 8, 
                  true + 9]

    
#     print("QUANTS: ", quants)

    if (pred >= true - quants[0]) and (pred <= true + quants[0]):
        r = 100
    elif (pred >= true - quants[1]) and (pred <= true + quants[1]):
        r = 90
    elif (pred >= true - quants[2]) and (pred <= true + quants[2]):
        r = 80
    elif (pred >= true - quants[3]) and (pred <= true + quants[3]):
        r = 70        
    elif (pred >= true - quants[4]) and (pred <= true + quants[4]):
        r = 60
    elif (pred >= true - quants[5]) and (pred <= true + quants[5]):
        r = 50        
    elif (pred >= true - quants[6]) and (pred <= true + quants[6]):
        r = 40        
    elif (pred >= true - quants[7]) and (pred <= true + quants[7]):
        r = 30
    elif (pred >= true - quants[8]) and (pred <= true + quants[8]):
        r = 20      
    else:
        r = 0
        
    return r


class Trainer:
    
    """A Recurrent Attention Model trainer.

    All hyperparameters are provided by the user in the
    config file.
    """

    def __init__(self, config, data_loader):
        
        """
        Construct a new Trainer instance.

        Args:
            config: object containing command line arguments.
            data_loader: A data iterator.
        """
        
        self.config = config

        if config.use_gpu and torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = torch.device("cpu")

        # glimpse network params
        self.patch_size = config.patch_size
        self.glimpse_scale = config.glimpse_scale
        self.num_patches = config.num_patches
        self.loc_hidden = config.loc_hidden
        self.glimpse_hidden = config.glimpse_hidden

        # core network params
        self.num_glimpses = config.num_glimpses
        self.hidden_size = config.hidden_size

        # reinforce params
        self.std = config.std
        self.M = config.M

        # data params
        if config.is_train:
            self.train_loader = data_loader[0]
            self.valid_loader = data_loader[1]
#             self.num_train = len(self.train_loader.sampler.indices)
#             self.num_valid = len(self.valid_loader.sampler.indices)
            self.num_train = len(self.train_loader)
            self.num_valid = len(self.valid_loader)

        else:
            self.test_loader = data_loader
            self.num_test = len(self.test_loader.dataset)
            
        self.num_classes = 4
        self.num_channels = 3

        # training params
        self.epochs = config.epochs
        self.start_epoch = 0
        self.momentum = config.momentum
        self.lr = config.init_lr

        # misc params
        self.best = config.best
        self.ckpt_dir = config.ckpt_dir
        self.logs_dir = config.logs_dir
        self.best_valid_acc = 9000000000000
        self.counter = 0
        self.lr_patience = config.lr_patience
        self.train_patience = config.train_patience
        self.use_tensorboard = config.use_tensorboard
        self.resume = config.resume
        self.print_freq = config.print_freq
        self.plot_freq = config.plot_freq
        self.model_name = "ram_{}_{}x{}_{}".format(
            config.num_glimpses,
            config.patch_size,
            config.patch_size,
            config.glimpse_scale,
        )

        self.plot_dir = "./plots/" + self.model_name + "/"
        if not os.path.exists(self.plot_dir):
            os.makedirs(self.plot_dir)

        # configure tensorboard logging
        if self.use_tensorboard:
            tensorboard_dir = self.logs_dir + self.model_name
            print("[*] Saving tensorboard logs to {}".format(tensorboard_dir))
            if not os.path.exists(tensorboard_dir):
                os.makedirs(tensorboard_dir)
            configure(tensorboard_dir)

        # build RAM model
        self.model = RecurrentAttention(
            self.patch_size,
            self.num_patches,
            self.glimpse_scale,
            self.num_channels,
            self.glimpse_hidden,
            self.loc_hidden,
            self.std,
            self.hidden_size,
            self.num_classes,
        )
        self.model.to(self.device)

        # initialize optimizer and scheduler
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.config.init_lr
        )
#         self.scheduler = ReduceLROnPlateau(
#             self.optimizer, "min", patience=self.lr_patience
#         )
        
        self.criterion = torch.nn.L1Loss(reduction = "mean")

    def reset(self):
        h_t = torch.zeros(
            self.batch_size,
            self.hidden_size,
            dtype=torch.float,
            device=self.device,
            requires_grad=True,
        )
        l_t = torch.FloatTensor(self.batch_size, 2).uniform_(-1, 1).to(self.device)
        l_t.requires_grad = True

        return h_t, l_t

    def train(self):
        """Train the model on the training set.

        A checkpoint of the model is saved after each epoch
        and if the validation accuracy is improved upon,
        a separate ckpt is created for use on the test set.
        """
        # load the most recent checkpoint
        if self.resume:
            self.load_checkpoint(best=False)

        print(
            "\n[*] Train on {} samples, validate on {} samples".format(
                self.num_train, self.num_valid
            )
        )

        for epoch in range(self.start_epoch, self.epochs):

            print(
                "\nEpoch: {}/{} - LR: {:.6f}".format(
                    epoch + 1, self.epochs, self.optimizer.param_groups[0]["lr"]
                )
            )

            # train for 1 epoch
            train_loss, train_mae = self.train_one_epoch(epoch)

            # evaluate on validation set
            valid_loss, val_mae  = self.validate(epoch)

            # reduce lr if validation loss plateaus
#             self.scheduler.step(-valid_loss)

            is_best = val_mae <  self.best_valid_acc
            msg1 = "train loss: {:.3f} - val loss: {:.3f}"
            msg2 = "- train mae: {:.3f} - val mae: {:.3f}"
            if is_best:
                self.counter = 0
                msg2 += " [*]"
            msg = msg1 + msg2
            print(
                msg.format(
                    train_loss, valid_loss, train_mae, val_mae
                )
            )
            
            
            with open("stats.txt", "r") as file1:
                prev = file1.read() + "\n"
                final = prev + msg.format(
                    train_loss, valid_loss, train_mae, val_mae
                )

            with open("stats.txt", "w") as file1:
                file1.write(final)               

                
            # check for improvement
            if not is_best:
                self.counter += 1
            if self.counter > self.train_patience:
                print("[!] No improvement in a while, stopping training.")
                return
            self.best_valid_acc = min(val_mae, self.best_valid_acc)
            self.save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "model_state": self.model.state_dict(),
                    "optim_state": self.optimizer.state_dict(),
                    "best_valid_acc": self.best_valid_acc,
                },
                is_best,
            )

    def train_one_epoch(self, epoch):
        
        """
        Train the model for 1 epoch of the training set.

        An epoch corresponds to one full pass through the entire
        training set in successive mini-batches.

        This is used by train() and should not be called manually.
        """
        
        self.model.train()
        batch_time = AverageMeter()
        losses = AverageMeter()
#         accs = AverageMeter()
        maes = AverageMeter()

        tic = time.time()
        with tqdm(total=self.num_train) as pbar:
            for i, (x, y_cont) in enumerate(self.train_loader):
                self.optimizer.zero_grad()

                x, y_cont = x.to(self.device), y_cont.to(self.device)

                plot = False
                if (epoch % self.plot_freq == 0) and (i == 0):
                    plot = True

                # initialize location vector and hidden state
                self.batch_size = x.shape[0]
                h_t, l_t = self.reset()

                # save images
                imgs = []
                imgs.append(x[0:9])

                # extract the glimpses
                locs = []
                log_pi = []
                baselines = []
                for t in range(self.num_glimpses - 1):
                    # forward pass through model
                    h_t, l_t, b_t, p = self.model(x, l_t, h_t)

                    # store
                    locs.append(l_t)
                    baselines.append(b_t)
                    log_pi.append(p)

                # last iteration
                h_t, l_t, b_t, p, cont_pred = self.model(x, l_t, h_t, last=True)
                
                log_pi.append(p)
                baselines.append(b_t)
                locs.append(l_t)

                # convert list to tensors and reshape
                try:
                    baselines = torch.stack(baselines).transpose(1, 0)
                except:
                    baselines = torch.stack(baselines).unsqueeze(1).transpose(1, 0)
                
                log_pi = torch.stack(log_pi).transpose(1, 0)

                # calculate reward
                R = torch.tensor([calc_reward(y_cont, cont_pred)], dtype = torch.float32).to(self.device)
                R = R.unsqueeze(1).repeat(1, self.num_glimpses)

                # compute losses for differentiable modules
                loss_baseline = F.mse_loss(baselines, R)
                
#                 print("R: ", R)
                
#                 print("baselines: ", baselines)
                
#                 print("loss_baseline: ", loss_baseline)
                
#                 print("\n")
                
                adjusted_reward = R - baselines.detach()
                loss_reinforce = torch.sum(-log_pi * adjusted_reward, dim=1)
                loss_reinforce = torch.mean(loss_reinforce, dim = 0)
                
                # Calculate the continuous MAE LOSS
                loss_action = self.criterion(cont_pred, y_cont.to(self.device))
                
                loss = loss_action + loss_baseline + loss_reinforce * 0.01

                # store
                losses.update(loss.item(), x.size()[0])
                maes.update(loss_action.item(), x.size()[0])

                # compute gradients and update SGD
                loss.backward()
                self.optimizer.step()

                # measure elapsed time
                toc = time.time()
                batch_time.update(toc - tic)

                pbar.set_description(
                    (
                        "{:.1f}s - loss: {:.3f} - mae: {:.3f}".format(
                            (toc - tic), loss.item(), loss_action.item()
                        )
                    )
                )
                pbar.update(self.batch_size)

                # dump the glimpses and locs
                if plot:
                    imgs = [g.cpu().data.numpy().squeeze() for g in imgs]
                    locs = [l.cpu().data.numpy() for l in locs]
                    pickle.dump(
                        imgs, open(self.plot_dir + "g_{}.p".format(epoch + 1), "wb")
                    )
                    pickle.dump(
                        locs, open(self.plot_dir + "l_{}.p".format(epoch + 1), "wb")
                    )

                # log to tensorboard
                if self.use_tensorboard:
                    iteration = epoch * len(self.train_loader) + i
                    log_value("train_loss", losses.avg, iteration)
                    log_value("train_acc", accs.avg, iteration)

            return losses.avg, maes.avg

    @torch.no_grad()
    def validate(self, epoch):
        
        """
        Evaluate the RAM model on the validation set.
        """
        
        losses = AverageMeter()
        accs = AverageMeter()
        maes = AverageMeter()

        for i, (x, y_cont) in enumerate(self.valid_loader):
            
            x, y_cont = x.to(self.device), y_cont.to(self.device)

            # duplicate M times
            x = x.repeat(self.M, 1, 1, 1)

            # initialize location vector and hidden state
            self.batch_size = x.shape[0]
            h_t, l_t = self.reset()

            # extract the glimpses
            log_pi = []
            baselines = []
            for t in range(self.num_glimpses - 1):
                # forward pass through model
                h_t, l_t, b_t, p = self.model(x, l_t, h_t)

                # store
                baselines.append(b_t)
                log_pi.append(p)

            # last iteration
            h_t, l_t, b_t, p, cont_pred = self.model(x, l_t, h_t, last = True)
            log_pi.append(p)
            baselines.append(b_t)

            # convert list to tensors and reshape
            # TRY AND EXCEPT TO ACCOUNT FOR A BATCH SIZE OF 1
            try:
                baselines = torch.stack(baselines).transpose(1, 0)
            except:
                baselines = torch.stack(baselines).unsqueeze(1).transpose(1, 0)
            log_pi = torch.stack(log_pi).transpose(1, 0)

            # average
            baselines = baselines.contiguous().view(self.M, -1, baselines.shape[-1])
            baselines = torch.mean(baselines, dim=0)

            log_pi = log_pi.contiguous().view(self.M, -1, log_pi.shape[-1])
            log_pi = torch.mean(log_pi, dim=0)

            # calculate reward
            R = torch.tensor([calc_reward(y_cont, cont_pred)], dtype = torch.float32).to(self.device)
            R = R.unsqueeze(1).repeat(1, self.num_glimpses)

            # compute losses for differentiable modules
            loss_baseline = F.mse_loss(baselines, R)
            loss_action = self.criterion(cont_pred, y_cont.to(self.device))

            adjusted_reward = R - baselines.detach()
            loss_reinforce = torch.sum(-log_pi * adjusted_reward, dim=1)
            loss_reinforce = torch.mean(loss_reinforce, dim=0)

            loss = loss_action + loss_baseline + loss_reinforce * 0.01

            # store
            losses.update(loss.item(), x.size()[0])
            maes.update(loss_action.item(), x.size()[0])

            # log to tensorboard
            if self.use_tensorboard:
                iteration = epoch * len(self.valid_loader) + i
                log_value("valid_loss", losses.avg, iteration)
                log_value("valid_acc", accs.avg, iteration)

        return losses.avg, maes.avg
    
    
    def save_checkpoint(self, state, is_best):
        """Saves a checkpoint of the model.

        If this model has reached the best validation accuracy thus
        far, a seperate file with the suffix `best` is created.
        """
        filename = self.model_name + "_ckpt.pth.tar"
        ckpt_path = os.path.join(self.ckpt_dir, filename)
        torch.save(state, ckpt_path)
        if is_best:
            filename = self.model_name + "_model_best.pth.tar"
            shutil.copyfile(ckpt_path, os.path.join(self.ckpt_dir, filename))

    def load_checkpoint(self, best=False):
        """Load the best copy of a model.

        This is useful for 2 cases:
        - Resuming training with the most recent model checkpoint.
        - Loading the best validation model to evaluate on the test data.

        Args:
            best: if set to True, loads the best model.
                Use this if you want to evaluate your model
                on the test data. Else, set to False in which
                case the most recent version of the checkpoint
                is used.
        """
        print("[*] Loading model from {}".format(self.ckpt_dir))

        filename = self.model_name + "_ckpt.pth.tar"
        if best:
            filename = self.model_name + "_model_best.pth.tar"
        ckpt_path = os.path.join(self.ckpt_dir, filename)
        ckpt = torch.load(ckpt_path)

        # load variables from checkpoint
        self.start_epoch = ckpt["epoch"]
        self.best_valid_acc = ckpt["best_valid_acc"]
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optim_state"])

        if best:
            print(
                "[*] Loaded {} checkpoint @ epoch {} "
                "with best valid acc of {:.3f}".format(
                    filename, ckpt["epoch"], ckpt["best_valid_acc"]
                )
            )
        else:
            print("[*] Loaded {} checkpoint @ epoch {}".format(filename, ckpt["epoch"]))

         
        
        

            
            
    @torch.no_grad()
    def predict(self, epoch, x, y_cont, checkpoint):

        """
        Evaluate the RAM model on the validation set.
        """
        
        self.model.load_state_dict(checkpoint)

        losses = AverageMeter()
        accs = AverageMeter()

        x, y_cont = x.to(self.device), y_cont.to(self.device)

        # duplicate M times
        x = x.repeat(self.M, 1, 1, 1)

        # initialize location vector and hidden state
        self.batch_size = x.shape[0]
        h_t, l_t = self.reset()

        # extract the glimpses
        log_pi = []
        baselines = []
        for t in range(self.num_glimpses - 1):
            
            # forward pass through model
            h_t, l_t, b_t, p = self.model(x, l_t, h_t)

            # store
            baselines.append(b_t)
            log_pi.append(p)

        # last iteration
        h_t, l_t, b_t, p, cont_pred = self.model(x, l_t, h_t, last=True)
                
        return cont_pred

    




    @torch.no_grad()
    def extract_locations(self, x, checkpoint):
        
        """
        Train the model for 1 epoch of the training set.

        An epoch corresponds to one full pass through the entire
        training set in successive mini-batches.

        This is used by train() and should not be called manually.
        """
        
        self.model.load_state_dict(checkpoint)

        x = x.to(self.device)

        # initialize location vector and hidden state
        self.batch_size = x.shape[0]
        h_t, l_t = self.reset()

        # extract the glimpses
        locs = []
        log_pi = []
        baselines = []
        for t in range(self.num_glimpses - 1):
            
            # forward pass through model
            h_t, l_t, b_t, p = self.model(x, l_t, h_t)

            # store
            locs.append(l_t[0:9])
            baselines.append(b_t)
            log_pi.append(p)

        # last iteration
        h_t, l_t, b_t, log_probas, p, cont_pred = self.model(x, l_t, h_t, last=True)
        
        locs.append(l_t[0:9])
        
        locs = torch.cat([locs[i].unsqueeze(0) for i in range(4)])
                
        return locs
            

