import os
from config.options import *
from config.utils import *
from config.learning_rates import get_learning_rate_scheduler
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.nn.functional as F
from torch.backends import cudnn
from scipy import stats
import numpy as np
from PIL import Image
import clip
from transformers import BertTokenizer
from transformers import LlamaTokenizer
from datasetllm import datasetllm
from models.MINTIQA_model import mintiqa
from config.options import *
def init_tokenizer():
    
    tokenizer = BertTokenizer.from_pretrained(opts.tokenizer)
    #tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenizer.add_special_tokens({'bos_token':'[DEC]'})
    tokenizer.add_special_tokens({'additional_special_tokens':['[ENC]']})       
    tokenizer.enc_token_id = tokenizer.additional_special_tokens_ids[0]  
    return tokenizer

def init_llm_tokenizer():
    llm_model = "/DATA/DATA2/wjr/vicuna-7b-v1.1"
    llm_tokenizer = LlamaTokenizer.from_pretrained(llm_model, use_fast=False, truncation_side="left")
    llm_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    llm_tokenizer.add_special_tokens({'bos_token': '</s>'})
    llm_tokenizer.add_special_tokens({'eos_token': '</s>'})
    llm_tokenizer.add_special_tokens({'unk_token': '</s>'})
    llm_tokenizer.padding_side = "right"
    return llm_tokenizer
    # self.llm_tokenizer.pad_token = self.llm_tokenizer.unk_token


def init_tokenizer2(truncation_side):
        tokenizer = BertTokenizer.from_pretrained(opts.tokenizer, truncation_side=truncation_side)
        tokenizer.add_special_tokens({"bos_token": "[DEC]"})

        return tokenizer
def texti_trans(text_instruct):
        tokenizer2 = init_tokenizer2(truncation_side="left")
        llm_tokenizer = init_llm_tokenizer()
        
        text_Qformer = tokenizer2(
                text_instruct,
                padding='max_length',
                truncation=True,
                max_length=256,
                return_tensors="pt",
            )
        
        llm_tokenizer.truncation_side = 'left'
        text_input_tokens = llm_tokenizer(
            text_instruct,
            return_tensors="pt",
            padding='max_length',
            truncation=True,
            max_length=256,
        )
        return text_Qformer.input_ids, text_Qformer.attention_mask, text_input_tokens.input_ids, text_input_tokens.attention_mask
def texto_trans(text_out):  
        llm_tokenizer = init_llm_tokenizer()
        llm_tokenizer.truncation_side = 'right'
        text_output_tokens = llm_tokenizer(
            text_out,
            return_tensors="pt",
            padding='max_length',
            truncation=True,
            max_length=256,
        )  
        return text_output_tokens.input_ids, text_output_tokens.attention_mask
if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #init_seeds(opts.seed)
    img_path ='/DATA/DATA2/wjr/allimg/img/1.png'
    prompt_ori = 'a corgi'
    tokenizer = init_tokenizer()
    model_path = '/DATA/DATA1/wjr/mintiqa/checkpoint/train2r_bs1_fix=0.6_lr=1e-05cosinesrcc/best_lr=1e-05.pt'
    state_dict = torch.load(model_path, map_location='cpu')
    model = mintiqa(device).to(device)
    msg = model.load_state_dict(state_dict,strict=False)
    print(msg)
    print("checkpoint loaded")
    # model.eval()
    # checkpoint =torch.load("/DATA/DATA2/wjr/instruct_blip_vicuna7b_trimmed.pth")
    # state_dict = checkpoint["model"]
    # for item in state_dict:
    #     print(item)
    # # msg = model2.blip2.load_state_dict(state_dict,strict=False)
    # msg = model.qformer2.load_state_dict(state_dict,strict=False)


    
    with torch.no_grad():  
        pil_image = Image.open(img_path)
        image = model.preprocess(pil_image)
        image = image.unsqueeze(0)
                
        text_input = tokenizer(prompt_ori, padding='max_length', truncation=True, max_length=35, return_tensors="pt")
        text_ids = text_input.input_ids
        text_mask = text_input.attention_mask
        text_ids = text_ids.view(text_ids.shape[0], -1).to(device) # [batch_size, seq_len]
        text_mask = text_mask.view(text_mask.shape[0], -1).to(device)
            
        prompt  = clip.tokenize(prompt_ori, truncate=True)
        prompt = prompt.view(prompt.shape[0], -1).to(device)
        img_better = image.to(device)
        text_instruct = "How is the correspondence of the image?"
        text_out = "The image quality is partially blurry, the image outline and content cannot be distinguished, the image details are partially missing."
        tq_ids,tq_mask,ti_ids,ti_mask= texti_trans(text_instruct)
        to_ids,to_mask = texto_trans(text_out)
        tq_ids = tq_ids.view(tq_ids.shape[0], -1).to(device) # [batch_size, seq_len]
        tq_mask = tq_mask.view(tq_ids.shape[0], -1).to(device) # [batch_size, seq_len]
        ti_mask = ti_mask.view(tq_ids.shape[0], -1).to(device) # [batch_size, seq_len]
        ti_ids = ti_ids.view(tq_ids.shape[0], -1).to(device) # [batch_size, seq_len]
        to_ids = to_ids.view(tq_ids.shape[0], -1).to(device) # [batch_size, seq_len]
        to_mask = to_mask.view(tq_ids.shape[0], -1).to(device) # [batch_size, seq_len]
        

       
        emb_better1, emb_better2, emb_better3, quality_vector = model.qformer1.forward_3score(image = img_better, text_ids=text_ids, text_mask=text_mask)
        # quality_vector =  model.blip2.forward_3score(image = img_better, text_ids=text_ids, text_mask=text_mask)
        # # # loss, back_vector =  model.blip2.forward_llm(image = img_better,vl_embeddings=quality_vector, tq_ids=tq_ids,tq_mask=tq_mask,ti_ids=ti_ids,ti_mask=ti_mask,to_ids=to_ids,to_mask=to_mask)
        Answer =  model.qformer2.generate(quality_vector, samples = {"image": img_better, "prompt": text_instruct}) 
        # quality_back = torch.cat([back_vector, quality_vector],dim=1)
        
        quality_score  = emb_better1.detach().cpu().numpy().item()
        authenticity_score  = emb_better2.detach().cpu().numpy().item()
        correspondence_score = emb_better3.detach().cpu().numpy().item()
        
        print('Quality:',quality_score,'\n')
        print('Authenticity:',authenticity_score,'\n')
        print('Correspondence:',correspondence_score,'\n')
        print('Answer:', Answer)