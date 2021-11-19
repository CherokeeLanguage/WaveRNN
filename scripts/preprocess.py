import os
from wavernn.utils.dsp import load_wav, normalize, float_2_label, encode_mu_law
from wavernn.utils import hparams as hp
import pickle
import numpy as np


def convert_file(path):
    y = load_wav(path)
    peak = np.abs(y).max()
    if hp.peak_norm or peak > 1.0:
        y /= peak
    if hp.voc_mode == 'RAW':
        quant = encode_mu_law(y, mu=2**hp.bits) if hp.mu_law else float_2_label(y, bits=hp.bits)
    elif hp.voc_mode == 'MOL':
        quant = float_2_label(y, bits=16)
    return quant.astype(np.int64)


def process_data(data_root, data_dirs, output_path):
    """
    Given language dependent directories and an output directory, 
    process wav files and save quantized wav and mel.
    """

    dataset = []
   
    c = 1
    for d in data_dirs:
        wav_d = os.path.join(data_root, d, "wavs")
        all_files = [os.path.splitext(f)[0] for f in os.listdir(wav_d)]

        for i, f in enumerate(all_files):
            file_id = '{:d}'.format(c).zfill(5)
            wav = convert_file(os.path.join(wav_d, f + ".wav"))
            mel = np.load(os.path.join(data_root, d, "gtas", f + ".npy"))
            mel = normalize(mel)
            np.save(os.path.join(output_path, "mel", file_id + ".npy"), mel, allow_pickle=False)
            np.save(os.path.join(output_path, "quant", file_id + ".npy"), wav, allow_pickle=False) 
            dataset.append((file_id, mel.shape[-1], os.path.basename(d)))
            c += 1
    
    # save dataset
    with open(os.path.join(output_path, 'dataset.pkl'), 'wb') as f:
        pickle.dump(dataset, f)
    
    print(f"Preprocessing done, total processed wav files: {c-1:,}")
    print(f"Processed files are located in:{os.path.abspath(output_path)}")


if __name__=="__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Preprocessing for WaveRNN and Tacotron')
    parser.add_argument("--base_directory", type=str, default=".", help="Base directory of the project.")
    parser.add_argument('--data_root', type=str, help='Directly point to dataset path (overrides hparams.data_path)')
    parser.add_argument("--inputs", nargs='+', type=str, help="Names of input directories.", required=True)
    parser.add_argument('--hp_file', type=str, default='hparams.py', help='The file to use for the hyperparameters')
    parser.add_argument("--output", type=str, help="Output directory (overrides hparams.data_path).")
    args = parser.parse_args()

    hp.configure(args.hp_file)
    if args.data_root is None:
        args.data_root = hp.data_path
    if args.output is None:
        args.output = hp.data_path

    output_dir = os.path.join(args.base_directory, args.output)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_quant_dir = os.path.join(args.base_directory, args.output, "quant")
    if not os.path.exists(output_quant_dir):
        os.makedirs(output_quant_dir)

    output_mel_dir = os.path.join(args.base_directory, args.output, "mel")
    if not os.path.exists(output_mel_dir):
        os.makedirs(output_mel_dir)

    process_data(args.data_root, args.inputs, output_dir)
