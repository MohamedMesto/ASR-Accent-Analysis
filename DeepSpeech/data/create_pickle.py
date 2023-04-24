import hydra
import torch
from tqdm import tqdm
from data.deepspeechpytorch.deepspeech_pytorch.loader.data_loader import SpectrogramDataset, AudioDataLoader
from data.deepspeechpytorch.deepspeech_pytorch.utils import load_model, load_decoder
from data.deepspeechpytorch.deepspeech_pytorch.configs.inference_config import EvalConfig
import gc
import os
from itertools import groupby
import pickle


cfg = EvalConfig
device = torch.device("cuda" if cfg.model.cuda else "cpu")

model = load_model(
    device=device,
    model_path=cfg.model.model_path
)
# orgnial solution, otherwise we will use the paper solution "from utils import load_model
test_dataset = SpectrogramDataset(
    audio_conf=model.spect_cfg,
    input_path=hydra.utils.to_absolute_path(cfg.test_path),
    labels=model.labels,
    normalize=True
)


test_loader = AudioDataLoader(
        test_dataset,
        batch_size=5, # cfg.batch_size,
        num_workers=4 # cfg.num_workers
    )





half = False ###  as an arrgument
for i, (data) in tqdm(enumerate(test_loader), total=len(test_loader)):

    inputs, targets, input_percentages, target_sizes, filenames = data # see if convert to variable and set requires_grad to true
    input_sizes = input_percentages.mul_(int(inputs.size(3))).int()
    inputs = inputs.to(device)
    if half:
        inputs = inputs.half()
    # unflatten targets
    split_targets = []
    offset = 0
    for size in target_sizes:
        split_targets.append(targets[offset:offset + size])
        offset += size

    inputs.requires_grad = True
    torch.set_grad_enabled(True)
    out, output_sizes, conv, rnn_0, rnn_1, rnn_2, rnn_3, rnn_4 = model(inputs, input_sizes)
    
    del input_percentages
    del conv
    del rnn_0
    del rnn_1 
    del rnn_2
    del rnn_3
    del rnn_4
    gc.collect()
    torch.cuda.empty_cache()
    chars, indices = torch.max(out,dim = 2)
    indices = indices.detach().cpu().numpy()
    target_path = '/workspace/attribution/grad/' 	

    for i in range(inputs.shape[0]): # batchsize
        if( os.path.exists(target_path + filenames[i] + '.pickle')):
            continue
        
        char_dict = {}
        grad_char_dict = {}
        
        group_list = [list(group) for k, group in groupby(indices[i][:output_sizes[i]])]
        combine_list = [0]
        idx_list = []
        for g in group_list:
            combine_list.append(combine_list[-1]+ len(g))
            idx_list.append(g[0])
        final = []
        del group_list
        for j in range(len(idx_list)):
            final.append(torch.sum(chars[i][combine_list[j]:combine_list[j+1]]))
        final_char = torch.stack(final)
        del combine_list		
        for k in range(len(idx_list)):
            
            if(idx_list[k] == 0):
                continue
            if(k!= len(idx_list)):
                final_char[k].backward(retain_graph = True)
            else:
                final_char[k].backward()
            inp_grad = inputs.grad[i].detach().clone().view(-1,inputs.shape[-1])
            #print(inp_grad.shape)
            model.zero_grad()
            inputs.grad.data.zero_()
            attr_grad = torch.norm(inp_grad, dim = 0)
            attr = torch.mul(inputs[i].view(-1,inputs.shape[-1]), inp_grad)
            attr = torch.sum(attr, dim = 0)
            attr_cpu = attr.detach().cpu().numpy()
            attr_grad_cpu = attr_grad.detach().cpu().numpy()
            char_dict[k] = attr_cpu
            grad_char_dict[k] = attr_grad_cpu
            del attr
            del inp_grad
            del attr_cpu
            del attr_grad_cpu
            gc.collect()
            torch.cuda.empty_cache()
            
        str_l = ["_'ABCDEFGHIJKLMNOPQRSTUVWXYZ "[m] for m in idx_list]
        str_op = ''.join(str_l)
        
        my_dict = {'output':str_op, 'attr dict':char_dict, 'grad_dict':grad_char_dict}
        with open('attribution/grad/{}.pickle'.format(filenames[i]),'wb+') as file:
            pickle.dump(my_dict, file)