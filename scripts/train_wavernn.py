import time
import numpy as np
import torch
from torch import optim
import torch.nn.functional as F
from wavernn.utils.display import stream
from wavernn.utils.dataset import get_vocoder_datasets
from wavernn.utils.distribution import discretized_mix_logistic_loss
from wavernn.utils import hparams as hp
from wavernn.models.fatchord_version import WaveRNN
from gen_wavernn import gen_testset
from wavernn.utils.paths import Paths
from wavernn.utils import data_parallel_workaround
from wavernn.utils.checkpoints import save_checkpoint, restore_checkpoint


def voc_train_loop(paths: Paths, model: WaveRNN, loss_func, optimizer, scheduler, train_set, test_set, total_steps):

    def get_lr(optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    device = next(model.parameters()).device

    total_iters = len(train_set)
    epochs = (total_steps - model.get_step()) // total_iters + 1

    for e in range(1, epochs + 1):
        start = time.time()
        running_loss = 0.

        for i, (x, y, m) in enumerate(train_set, 1):
            x, m, y = x.to(device), m.to(device), y.to(device)

            # Parallelize model onto GPUS using workaround due to python bug
            if device.type == 'cuda' and torch.cuda.device_count() > 1:
                y_hat = data_parallel_workaround(model, x, m)
            else:
                y_hat = model(x, m)

            if model.mode == 'RAW':
                y_hat = y_hat.transpose(1, 2).unsqueeze(-1)

            elif model.mode == 'MOL':
                y = y.float()

            y = y.unsqueeze(-1)

            loss = loss_func(y_hat, y)

            optimizer.zero_grad()
            loss.backward()
            if hp.voc_clip_grad_norm is not None:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), hp.voc_clip_grad_norm)
                if torch.isnan(grad_norm).any():
                    print('grad_norm was NaN!')
            optimizer.step()

            running_loss += loss.item()
            avg_loss = running_loss / i

            speed = i / (time.time() - start)

            step = model.get_step()
            k = step // 1000

            if step % hp.voc_checkpoint_every == 0:
                gen_testset(model, test_set, hp.voc_gen_at_checkpoint, hp.voc_gen_batched,
                            hp.voc_target, hp.voc_overlap, paths.voc_output)
                ckpt_name = f'wave_step{k}K'
                save_checkpoint(paths, model, optimizer, scheduler, name=ckpt_name, is_silent=True)

            msg = f'| Epoch: {e}/{epochs} ({i}/{total_iters}) | Loss: {avg_loss:.4f} | LR: {get_lr(optimizer):.4f} | {speed:.1f} steps/s | Step: {k}k | '
            stream(msg)

        # Must save latest optimizer state to ensure that resuming training
        # doesn't produce artifacts
        save_checkpoint(paths, model, optimizer, scheduler, is_silent=True)
        model.log(paths.voc_log, msg)
        print(' ')

        if hp.lr_decay_start - hp.lr_decay_each < e * total_iters:
            scheduler.step()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train WaveRNN Vocoder')
    parser.add_argument('--force_train', '-f', action='store_true', help='Forces the model to train past total steps')
    parser.add_argument('--hp_file', metavar='FILE', default='hparams.py', help='The file to use for the hyperparameters')
    args = parser.parse_args()

    hp.configure(args.hp_file)  
    batch_size = hp.voc_batch_size
    
    paths = Paths(hp.data_path, hp.voc_model_id)

    if torch.cuda.is_available():
        device = torch.device('cuda')
        if batch_size % torch.cuda.device_count() != 0:
            raise ValueError('`batch_size` must be evenly divisible by n_gpus!')
    else:
        device = torch.device('cpu')

    print('\nInitialising Model...\n')

    # Instantiate WaveRNN Model
    voc_model = WaveRNN(rnn_dims=hp.voc_rnn_dims,
                        fc_dims=hp.voc_fc_dims,
                        bits=hp.bits,
                        pad=hp.voc_pad,
                        upsample_factors=hp.voc_upsample_factors,
                        feat_dims=hp.num_mels,
                        compute_dims=hp.voc_compute_dims,
                        res_out_dims=hp.voc_res_out_dims,
                        res_blocks=hp.voc_res_blocks,
                        hop_length=hp.hop_length,
                        sample_rate=hp.sample_rate,
                        mode=hp.voc_mode).to(device)

    # Check to make sure the hop length is correctly factorised
    assert np.cumprod(hp.voc_upsample_factors)[-1] == hp.hop_length

    train_set, test_set = get_vocoder_datasets(paths.data, batch_size)

    optimizer = torch.optim.Adam(voc_model.parameters(), lr=hp.voc_lr, weight_decay=hp.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, hp.lr_decay_each // len(train_set), gamma=hp.lr_decay)
    
    restore_checkpoint(paths, voc_model, optimizer, scheduler, create_if_missing=True)
  
    total_steps = 10_000_000 if args.force_train else hp.voc_total_steps
    loss_func = F.cross_entropy if voc_model.mode == 'RAW' else discretized_mix_logistic_loss  
    voc_train_loop(paths, voc_model, loss_func, optimizer, scheduler, train_set, test_set, total_steps)
