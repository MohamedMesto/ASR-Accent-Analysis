# print('No God But ALLAH')
import hydra
import torch

from data.deepspeechpytorch.deepspeech_pytorch.decoder import GreedyDecoder
from data.deepspeechpytorch.deepspeech_pytorch.loader.data_loader import SpectrogramDataset, AudioDataLoader
from data.deepspeechpytorch.deepspeech_pytorch.utils import load_model, load_decoder
from data.deepspeechpytorch.deepspeech_pytorch.validation import run_evaluation


@torch.no_grad()
# def evaluate_2(cfg: EvalConfig):
def evaluate_2(cuda,model_path,decoder,manifest_filepath,
               batch_size):
    device = torch.device("cuda" if cuda else "cpu")

    # model = load_model(
    #     device=device,
    #     model_path=model.model_path
    # )

    model = load_model(
        device=device,
        model_path=model_path,
        half=True
    )
    # decoder = load_decoder(
    #     labels=model.labels,
    #     cfg=cfg.lm
    # )
    decoder = decoder


    target_decoder = GreedyDecoder(
        labels=model.labels,
        blank_index=model.labels.index('_')
    )
    # test_dataset = SpectrogramDataset(
    #     audio_conf=model.spect_cfg,
    #     input_path=hydra.utils.to_absolute_path(cfg.test_path),
    #     labels=model.labels,
    #     normalize=True
    # )

    test_dataset = SpectrogramDataset(
        audio_conf=model.audio_conf, # could be changed
        manifest_filepath=manifest_filepath,
        labels=model.labels,
        normalize=True
    )


    # test_loader = AudioDataLoader(
    #     test_dataset,
    #     batch_size=cfg.batch_size,
    #     num_workers=cfg.num_workers
    # )

    test_loader = AudioDataLoader(
        test_dataset,
        batch_size=batch_size,
        # num_workers=num_workers
        num_workers=4
    )

    # wer, cer = run_evaluation(
    #     test_loader=test_loader,
    #     device=device,
    #     model=model,
    #     decoder=decoder,
    #     target_decoder=target_decoder,
    #     precision=cfg.model.precision
    # )

    wer, cer = run_evaluation(
        test_loader=test_loader,
        device=device,
        model=model,
        decoder=decoder,
        target_decoder=target_decoder,
        precision=model.precision # could be changed
    )
    print('Test Summary \t'
          'Average WER {wer:.3f}\t'
          'Average CER {cer:.3f}\t'.format(wer=wer, cer=cer))

    return wer, cer