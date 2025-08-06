import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from config.options import *
from config.utils import *
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from models.MINT_Qformers import *
from transformers import BertTokenizer
from config.options import *
import clip

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
    #tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenizer.add_special_tokens({'bos_token':'[DEC]'})
    tokenizer.add_special_tokens({'additional_special_tokens':['[ENC]']})       
    tokenizer.enc_token_id = tokenizer.additional_special_tokens_ids[0]  
    return tokenizer


class mintiqa(nn.Module):
    def __init__(self, device='cpu'):
        super().__init__()
        #self.preprocess = _transform(config['BLIP']['image_size'])
        self.tokenizer = init_tokenizer()
        self.device = device
        self.qformer1 = MINTqformer1()
        # self.qformer2 = MINTqformer2()
        checkpoint =torch.load("/DATA/DATA3/wjr/weight/blip2_pretrained.pth")
        state_dict = checkpoint["model"]
        msg = self.qformer1.load_state_dict(state_dict,strict=False)
        print(msg)
        # for name, parms in self.qformer1.named_parameters():
        #     parms.requires_grad_(True)
        # msg = self.qformer2.load_state_dict(state_dict,strict=False)

        # checkpoint =torch.load("/DATA/DATA2/wjr/instruct_blip_vicuna7b_trimmed.pth")
        # state_dict = checkpoint["model"]
        # for item in state_dict:
        #     print(item)
        
        # msg = self.qformer2.load_state_dict(state_dict,strict=False)
        # print(msg)
        self.preprocess = _transform(config['BLIP']['image_size'])
        # for name, parms in self.qformer1.named_parameters():
        #     parms.requires_grad_(False)
        # checkpoint =torch.load("/home/wangjiarui/LAVIS/weight/instruct_blip_vicuna7b_trimmed.pth")
        # state_dict = checkpoint["model"]
        # msg2 = self.blip2.load_state_dict(state_dict,strict=False)
        # #print(msg)
        # print(msg2)
        
    def forward(self, batch_data):

        batch_data = self.encode_pair(batch_data)
        lossllm = batch_data['lossllm']
        
        emb_better1 = batch_data['emb_better1']
        reward_better1 = emb_better1
        reward_better1 = reward_better1[:,None]
        reward_worse1 = batch_data['emb_worse1']
        reward_worse1 = reward_worse1[:,None]
        reward1 = torch.concat((reward_better1, reward_worse1), dim=1)
       
        # emb_better2 = batch_data['emb_better2']
        # reward_better2 = emb_better2
        # reward_better2 = reward_better2[:,None]
        # reward_worse2 = batch_data['emb_worse2']
        # reward_worse2 = reward_worse2[:,None]
        # reward2 = torch.concat((reward_better2, reward_worse2), dim=1)

        # emb_better3 = batch_data['emb_better3']
        # reward_better3 = emb_better3
        # reward_better3 = reward_better3[:,None]
        # reward_worse3 = batch_data['emb_worse3']
        # reward_worse3 = reward_worse3[:,None]
        # reward3 = torch.concat((reward_better3, reward_worse3), dim=1)

        return reward1, lossllm
        
    def encode_pair(self, batch_data):
        text_ids = batch_data['text_ids']
        tq_ids = batch_data['tq_ids']
        tq_mask = batch_data['tq_mask']
        ti_ids = batch_data['ti_ids']
        ti_mask = batch_data['ti_mask']
        to_ids = batch_data['to_ids']
        to_mask = batch_data['to_mask']
        text_mask = batch_data['text_mask']
        img_better = batch_data['img']
        text_ids = text_ids.view(text_ids.shape[0], -1).to(self.device) # [batch_size, seq_len]
        text_mask = text_mask.view(text_mask.shape[0], -1).to(self.device) # [batch_size, seq_len]
        tq_ids = tq_ids.view(tq_ids.shape[0], -1).to(self.device) # [batch_size, seq_len]
        tq_mask = tq_mask.view(tq_mask.shape[0], -1).to(self.device) # [batch_size, seq_len]
        ti_ids = ti_ids.view(ti_ids.shape[0], -1).to(self.device) # [batch_size, seq_len]
        ti_mask = ti_mask.view(ti_mask.shape[0], -1).to(self.device) # [batch_size, seq_len]
        to_ids = to_ids.view(to_ids.shape[0], -1).to(self.device) # [batch_size, seq_len]
        to_mask = to_mask.view(to_mask.shape[0], -1).to(self.device)#prompt = prompt
        img_better = img_better.to(self.device) # [batch_size, C, H, W]
        emb_worse1 = batch_data['moz1']
        emb_worse1 = emb_worse1.to(self.device)
        # emb_worse2 = batch_data['moz2']
        # emb_worse2 = emb_worse2.to(self.device)
        # emb_worse3 = batch_data['moz3']
        # emb_worse3 = emb_worse3.to(self.device)
        emb_better1, emb_better2, emb_better3, quality_vector = self.qformer1.forward_3score(image = img_better, text_ids=text_ids, text_mask=text_mask)
        # lossllm, back_vector = self.qformer2.forward_llm(image = img_better,vl_embeddings=quality_vector, tq_ids=tq_ids,tq_mask=tq_mask,ti_ids=ti_ids,ti_mask=ti_mask,to_ids=to_ids,to_mask=to_mask)
        
        # emb_better1, emb_better2, emb_better3 = self.score.forward_3score(image = img_better, text_ids=text_ids, text_mask=text_mask)
        # loss = self.score.forward_llm(image = img_better,target_score1 = emb_worse1, target_score2 = emb_worse2, target_score3 = emb_worse3,tq_ids=tq_ids,tq_mask=tq_mask,ti_ids=ti_ids,ti_mask=ti_mask,to_ids=to_ids,to_mask=to_mask)
        lossllm = 0
        batch_data = {
            'emb_better1': emb_better1,
            'emb_better2': emb_better2,
            'emb_better3': emb_better3,
            'emb_worse1': emb_worse1,
            # 'emb_worse2': emb_worse2,
            # 'emb_worse3': emb_worse3,
            'lossllm': lossllm,
        }
        return batch_data