import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from models.MINT_Qformers import *
from transformers import BertTokenizer
from config.options_infer import *

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def _transform(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

def init_tokenizer():
    
    tokenizer = BertTokenizer.from_pretrained(opts.tokenizer)
    tokenizer.add_special_tokens({'bos_token':'[DEC]'})
    tokenizer.add_special_tokens({'additional_special_tokens':['[ENC]']})       
    tokenizer.enc_token_id = tokenizer.additional_special_tokens_ids[0]  
    return tokenizer


class mintiqa(nn.Module):
    def __init__(self, device='cpu'):
        super().__init__()
        self.tokenizer = init_tokenizer()
        self.device = device
        self.qformer1 = MINTqformer1()
        self.qformer2 = MINTqformer2()
        self.preprocess = _transform(224)
    def forward(self, batch_data):

        batch_data = self.encode_pair(batch_data)
        lossllm = batch_data['lossllm']
        
        score1 = batch_data['score1'][:,None]
        mos1 = batch_data['mos1'][:,None]
        reward1 = torch.concat((score1, mos1), dim=1)
       
        score2 = batch_data['score2'][:,None]
        mos2 = batch_data['mos2'][:,None]
        reward2 = torch.concat((score2, mos2), dim=1)

        score3 = batch_data['score3'][:,None]
        mos3 = batch_data['mos3'][:,None]
        reward3 = torch.concat((score3, mos3), dim=1)

        return reward1, reward2, reward3, lossllm
        
    def encode_pair(self, batch_data):
        text_ids = batch_data['text_ids']
        text_mask = batch_data['text_mask']
        img = batch_data['img']
        text_ids = text_ids.view(text_ids.shape[0], -1).to(self.device) # [batch_size, seq_len]
        text_mask = text_mask.view(text_mask.shape[0], -1).to(self.device) # [batch_size, seq_len]
        tq_ids = tq_ids.view(tq_ids.shape[0], -1).to(self.device) # [batch_size, seq_len]
        tq_mask = tq_mask.view(tq_mask.shape[0], -1).to(self.device) # [batch_size, seq_len]
        ti_ids = ti_ids.view(ti_ids.shape[0], -1).to(self.device) # [batch_size, seq_len]
        ti_mask = ti_mask.view(ti_mask.shape[0], -1).to(self.device) # [batch_size, seq_len]
        to_ids = to_ids.view(to_ids.shape[0], -1).to(self.device) # [batch_size, seq_len]
        to_mask = to_mask.view(to_mask.shape[0], -1).to(self.device)#prompt = prompt
        img = img.to(self.device) # [batch_size, C, H, W]
        mos1 = batch_data['moz1'].to(self.device)
        mos2 = batch_data['moz2'].to(self.device)
        mos3 = batch_data['moz3'].to(self.device)
        score1, score2, score3, quality_vector = self.qformer1.forward_3score(image = img, text_ids=text_ids, text_mask=text_mask)
        lossllm, back_vector = self.qformer2.forward_llm(image = img,vl_embeddings=quality_vector, tq_ids=tq_ids,tq_mask=tq_mask,ti_ids=ti_ids,ti_mask=ti_mask,to_ids=to_ids,to_mask=to_mask)
        batch_data = {
            'score1': score1,
            'score2': score2,
            'score3': score3,
            'mos1': mos1,
            'mos2': mos2,
            'mos3': mos3,
            'lossllm': lossllm,
        }
        return batch_data