import os
import json
import math
import torch
from torch.utils.data import Dataset
from config.utils import *
from config.options import *
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from tqdm import tqdm
from transformers import BertTokenizer
from config.options import *
import clip
from transformers import LlamaTokenizer

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
def init_llm_tokenizer():
    llm_model = opts.llm
    llm_tokenizer = LlamaTokenizer.from_pretrained(llm_model, use_fast=False, truncation_side="left")
    llm_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    llm_tokenizer.add_special_tokens({'bos_token': '</s>'})
    llm_tokenizer.add_special_tokens({'eos_token': '</s>'})
    llm_tokenizer.add_special_tokens({'unk_token': '</s>'})
    llm_tokenizer.padding_side = "right"
    return llm_tokenizer
    # self.llm_tokenizer.pad_token = self.llm_tokenizer.unk_token


def init_tokenizer():
    
    tokenizer = BertTokenizer.from_pretrained(opts.tokenizer)
    #tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenizer.add_special_tokens({'bos_token':'[DEC]'})
    tokenizer.add_special_tokens({'additional_special_tokens':['[ENC]']})       
    tokenizer.enc_token_id = tokenizer.additional_special_tokens_ids[0]  
    return tokenizer

def init_tokenizer2(truncation_side):
        tokenizer = BertTokenizer.from_pretrained(opts.tokenizer, truncation_side=truncation_side)
        tokenizer.add_special_tokens({"bos_token": "[DEC]"})
        return tokenizer

class datasetllm(Dataset):
    def __init__(self, dataset):
        self.preprocess = _transform(config['BLIP']['image_size'])
        self.tokenizer = init_tokenizer()
        self.tokenizer2 = init_tokenizer2(truncation_side="left")
        self.llm_tokenizer = init_llm_tokenizer()
        
        if dataset == "train":
            with open(opts.train_file, "r") as f:
                self.data = json.load(f)
        if dataset == "valid":
            with open(opts.test_file, "r") as f:
                self.data = json.load(f)
        if dataset == "test":
            with open(opts.test_file, "r") as f:
                self.data = json.load(f)


        self.prompts, self.img_set, self.mos1, self.mos2, self.mos3, self.text_instruct, self.text_out = self.make_data()


        self.iters_per_epoch = int(math.ceil(len(self.data)*1.0/opts.batch_size))

    def __getitem__(self, index):
        item = {}
        prompt = self.prompts[index]
        imgpath = self.img_set[index]
        text_instruct = self.text_instruct[index]
        text_out = self.text_out[index]
        item['img'] = self._img_trans(imgpath)
        item['text_ids'] , item['text_mask'] = self._txt_trans(prompt)
        item['tq_ids'], item['tq_mask'], item['ti_ids'], item['ti_mask']= self._texti_trans(text_instruct)
        item['to_ids'], item['to_mask'] = self._texto_trans(text_out)
        item['moz1'] = self.mos1[index]
        item['moz2'] = self.mos2[index]
        item['moz3'] = self.mos3[index]

        return item

    def __len__(self):
        return len(self.data)
    
    def store_dataset(self, dataset):
        makedir(config['pair_store_base'])
        torch.save(self.data, os.path.join(config['pair_store_base'], f"{dataset}.pth"))
    

    def _texti_trans(self, text_instruct):
        
        text_Qformer = self.tokenizer2(
                text_instruct,
                padding='max_length',
                truncation=True,
                max_length=256,
                return_tensors="pt",
            )
        self.llm_tokenizer.padding_side = "right"
        self.llm_tokenizer.truncation_side = 'left'
        text_input_tokens = self.llm_tokenizer(
            text_instruct,
            return_tensors="pt",
            padding='max_length',
            truncation=True,
            max_length=256,
        )
        return text_Qformer.input_ids, text_Qformer.attention_mask, text_input_tokens.input_ids, text_input_tokens.attention_mask

    def _texto_trans(self, text_out):  
        self.llm_tokenizer.padding_side = "right"
        self.llm_tokenizer.truncation_side = 'right'
        text_output_tokens = self.llm_tokenizer(
            text_out+self.llm_tokenizer.eos_token,
            return_tensors="pt",
            padding='max_length',
            truncation=True,
            max_length=256,
        )  
        return text_output_tokens.input_ids, text_output_tokens.attention_mask


    def _img_trans(self, imgpath):
        
        pil_image = Image.open(imgpath)
        image = self.preprocess(pil_image)
        return image

    def _txt_trans(self, text):
        
        text_input = self.tokenizer(text, padding='max_length', truncation=True, max_length=256, return_tensors="pt")
        return text_input.input_ids, text_input.attention_mask
    
    def make_data(self):
        
        img_set = []
        prompts = []
        mos1 = []
        mos2 = []
        mos3 = []
        a1 = []
        q1 = []

        
        bar = tqdm(range(len(self.data)), desc=f'making dataset: ')
        for item in self.data:
            
            item['q1'] = ""
            item['a1'] = ""
            img_path = os.path.join(opts.imgpath, item['img'])
            
            # pil_image = Image.open(img_path)
            # image = self.preprocess(pil_image)
            img_set.append(img_path)
            prompts.append(item["prompt"])
            mos1.append(item['moz1'])
            mos2.append(item['moz2'])
            mos3.append(item['moz3'])
            q1.append(item['q1'])
            a1.append(item['a1'])
 
            bar.update(1)

        return prompts, img_set, mos1, mos2, mos3, q1, a1 
