"""
Requires Transformer 4.28 and above, implementation may change according the Llama implementation
"""
import logging
import string
from packaging import version

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn
from torch.nn import functional as F
import transformers
from lavis.models.base_model import all_gather_with_grad, concat_all_gather
from lavis.common.registry import registry
from config.options import *
# from lavis.models.blip2_models.blip2 import Blip2Base, disabled_train
from models.blip2r import Blip2Base, disabled_train
#@registry.register_model("blip2_vicuna_instruct")
class Blip2wangjiaruitrain2(Blip2Base):
    """
    BLIP2 Vicuna model.
    Supported model types:
        - vicuna7b
        - vicuna13b
    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip2_vicuna_instruct", "vicuna7b")
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "vicuna7b": "configs/models/blip2/blip2_instruct_vicuna7b.yaml",
        "vicuna13b": "configs/models/blip2/blip2_instruct_vicuna13b.yaml",
    }

    def __init__(
        self,
        vit_model="eva_clip_g",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp32",
        freeze_vit=True,
        num_query_token=32,
        embed_dim=256,
        cross_attention_freq=2,
        llm_model="vicuna7b",
        prompt="",
        max_txt_len=128,
        max_output_txt_len=256,
        apply_lemmatizer=False,
        qformer_text_input=True,
        use_nucleus_sampling=False,
        num_beams=5,
        max_length=256,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.5,
        length_penalty=1,
        num_captions=1,
        temperature=1,
    ):
        super().__init__()
        transformers_version = version.parse(transformers.__version__)
        print(transformers_version)
        #assert transformers_version >= version.parse("4.28"), "BLIP-2 Vicuna requires transformers>=4.28"        
        from transformers import LlamaTokenizer
        #from lavis.models.blip2_models.modeling_llama import LlamaForCausalLM
        from models.modeling_llama import LlamaForCausalLM
        
        self.tokenizer = self.init_tokenizer(truncation_side="left")

        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )

        self.image_layer_num = 38
        image_fix_num = "blocks.{}".format(int(self.image_layer_num * 0.7))
    
        for name, parms in self.visual_encoder.named_parameters():
            parms.requires_grad_(False)
            if image_fix_num in name:
                break


        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            logging.info("freeze vision encoder")

        self.Qformer1, self.query_tokens1 = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features
        )
        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features
        )
        self.query_tokens_initial = self.query_tokens  

        self.Qformer1.resize_token_embeddings(len(self.tokenizer))
        self.Qformer.resize_token_embeddings(len(self.tokenizer))
        state_dict = self.Qformer1.state_dict()
        for name, param in self.Qformer1.named_parameters():
            if "_query" in name:
                key_orig = name.replace("_query", "")
                param.data.copy_(state_dict[key_orig])
        state_dict = self.Qformer.state_dict()
        for name, param in self.Qformer.named_parameters():
            if "_query" in name:
                key_orig = name.replace("_query", "")
                param.data.copy_(state_dict[key_orig])

        self.vision_proj = nn.Linear(self.Qformer1.config.hidden_size, embed_dim)
        self.text_proj = nn.Linear(self.Qformer1.config.hidden_size, embed_dim)

        self.quality1 = self.quality_regression(self.Qformer1.config.hidden_size, 48, 3)
        self.quality2 = self.quality_regression(self.Qformer1.config.hidden_size, 48, 3)
        self.quality3 = self.quality_regression(self.Qformer1.config.hidden_size, 48, 3)

        self.itm_head = nn.Linear(self.Qformer1.config.hidden_size, 2)
        self.itm_head2 = nn.Linear(self.Qformer1.config.hidden_size, 1)
        self.itm_head3 = nn.Linear(self.Qformer1.config.hidden_size, 3)
        print('self.Qformer.config.hidden_size',self.Qformer1.config.hidden_size)

        self.temp = nn.Parameter(0.07 * torch.ones([]))

        self.max_txt_len = max_txt_len

        #self.max_txt_len = max_txt_len

        # if not qformer_text_input:
        #     self.Qformer.bert.embeddings.word_embeddings = None
        #     self.Qformer.bert.embeddings.position_embeddings = None
        #     for layer in self.Qformer.bert.encoder.layer:
        #         layer.output = None
        #         layer.intermediate = None
        # else:
        #     self.Qformer.resize_token_embeddings(len(self.tokenizer))
        self.Qformer.cls = None

        #self.llm_tokenizer = LlamaTokenizer.from_pretrained(llm_model, use_fast=False, truncation_side="left")
        # self.llm_model = LlamaForCausalLM.from_pretrained(
        #     llm_model, torch_dtype=torch.float16
        # )
        llm_model = opts.llm
        self.llm_tokenizer = LlamaTokenizer.from_pretrained(llm_model, use_fast=False, truncation_side="left")
        self.llm_model = LlamaForCausalLM.from_pretrained(
            llm_model, torch_dtype=torch.float16
        )
        self.llm_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.llm_tokenizer.add_special_tokens({'bos_token': '</s>'})
        self.llm_tokenizer.add_special_tokens({'eos_token': '</s>'})
        self.llm_tokenizer.add_special_tokens({'unk_token': '</s>'})
        # self.llm_tokenizer.pad_token = self.llm_tokenizer.unk_token

        self.llm_model.resize_token_embeddings(len(self.llm_tokenizer))

        # self.eos_token_id = self.llm_tokenizer(
        #     self.llm_tokenizer.eos_token, add_special_tokens=False
        # ).input_ids[0]

        for name, param in self.llm_model.named_parameters():
            param.requires_grad = False

        self.llm_proj = nn.Linear(
            self.Qformer.config.hidden_size, self.llm_model.config.hidden_size
        )

        self.max_txt_len = max_txt_len
        self.max_output_txt_len = max_output_txt_len
        self.prompt = prompt
        prompt_tokens = self.llm_tokenizer(self.prompt, return_tensors="pt")
        self.prompt_length = prompt_tokens.attention_mask.sum(1)

        self._lemmatizer = None

        self.qformer_text_input = qformer_text_input
        self.zero_forward = self.make_zero_conv(32)
        # self.quality1 = self.quality_regression(self.Qformer.config.hidden_size, 48, 3)
        # self.quality2 = self.quality_regression(self.Qformer.config.hidden_size, 48, 3)
        # self.quality3 = self.quality_regression(self.Qformer.config.hidden_size, 48, 3)

        # self.itm_head = nn.Linear(self.Qformer.config.hidden_size, 2)
        # self.itm_head2 = nn.Linear(self.Qformer.config.hidden_size, 1)
        # self.itm_head3 = nn.Linear(self.Qformer.config.hidden_size, 3)

    def quality_regression(self,in_channels, middle_channels, out_channels):
        regression_block = nn.Sequential(
            nn.Linear(in_channels, middle_channels),
            nn.Linear(middle_channels, out_channels),          
        )

        return regression_block

    def zero_module(self, module):
        """
        Zero out the parameters of a module and return it.
        """
        for p in module.parameters():
            p.detach().zero_()
        return module

    def make_zero_conv(self, channels=32):
        #return zero_module(conv_nd(self.dims, channels, channels, 1, padding=0))
        return self.zero_module(nn.Conv1d(channels, channels, 1, padding=0))

# def make_zero_conv(self, channels):
#         return TimestepEmbedSequential(zero_module(conv_nd(self.dims, channels, channels, 1, padding=0)))

    

    def concat_text_input_output(self, input_ids, input_atts, output_ids, output_atts):
        input_part_targets_len = []
        llm_tokens = {"input_ids": [], "attention_mask": []}
        for i in range(input_ids.size(0)):
            this_input_ones = input_atts[i].sum()
            input_part_targets_len.append(this_input_ones)
            llm_tokens['input_ids'].append(
                torch.cat([
                    input_ids[i][:this_input_ones],
                    output_ids[i][1:],
                    input_ids[i][this_input_ones:]
                ])
            )
            llm_tokens['attention_mask'].append(
                torch.cat([
                    input_atts[i][:this_input_ones],
                    output_atts[i][1:],
                    input_atts[i][this_input_ones:]
                ])
            )
        llm_tokens['input_ids'] = torch.stack(llm_tokens['input_ids'])
        llm_tokens['attention_mask'] = torch.stack(llm_tokens['attention_mask'])
        return llm_tokens, input_part_targets_len

    def forward_3score(self, image, text_ids, text_mask):
        # image = samples["image"]
        # text = samples["text_input"]

        self.image_embeds = self.ln_vision(self.visual_encoder(image))
        self.image_atts = torch.ones(self.image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )
        query_tokens = self.query_tokens.expand(self.image_embeds.shape[0], -1, -1)

        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=self.image_embeds,
            encoder_attention_mask=self.image_atts,
            use_cache=True,
            return_dict=True,
        )

        image_feats = F.normalize(
            self.vision_proj(query_output.last_hidden_state), dim=-1
        )
        text_output = self.Qformer.bert(
            text_ids,
            attention_mask=text_mask,
            return_dict=True,
        )
        text_feat = F.normalize(
            self.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1
        )
         ###============== Image-text Contrastive ===================###
        image_feats_all = concat_all_gather(
            image_feats
        )  # [batch_size*num_gpu, num_query_tokens, embed_dim]
        text_feat_all = concat_all_gather(text_feat)  # [batch_size*num_gpu, embed_dim]

        sim_q2t = torch.matmul(
            image_feats.unsqueeze(1), text_feat_all.unsqueeze(-1)
        ).squeeze()
        # [batch_size, batch_size*num_gpu, num_query_tokens]

        # image-text similarity: aggregate across all query tokens
        sim_i2t, _ = sim_q2t.max(-1)
        sim_i2t = sim_i2t / self.temp

        # text-query similarity: [batch_size, batch_size*num_gpu, num_query_tokens]
        sim_t2q = torch.matmul(
            text_feat.unsqueeze(1).unsqueeze(1), image_feats_all.permute(0, 2, 1)
        ).squeeze()

        # text-image similarity: aggregate across all query tokens
        sim_t2i, _ = sim_t2q.max(-1)
        sim_t2i = sim_t2i / self.temp  # [batch_size, batch_size*num_gpu]

        # rank = dist.get_rank()
        rank = 0
        bs = image.size(0)
        targets = torch.linspace(rank * bs, rank * bs + bs - 1, bs, dtype=int).to(
            image.device
        )

        query_tokens_itm = self.query_tokens1.expand(text_ids.shape[0], -1, -1)
        query_atts_itm = torch.ones(query_tokens_itm.size()[:-1], dtype=torch.long).to(
            image.device
        )
        attention_mask_all = torch.cat([query_atts_itm, text_mask], dim=1)

        output_itm = self.Qformer1.bert(
            text_ids,
            query_embeds=query_tokens_itm,
            attention_mask=attention_mask_all,
            encoder_hidden_states=self.image_embeds,
            encoder_attention_mask=self.image_atts,
            return_dict=True,
        )

        vl_embeddings = output_itm.last_hidden_state[:, : query_tokens_itm.size(1), :]
        itm_logit1 = self.quality1(vl_embeddings)
        itm_logit1 = itm_logit1[:, :, 1].mean(dim=1)
        itm_logit2 = self.quality2(vl_embeddings)
        itm_logit2 = itm_logit2[:, :, 1].mean(dim=1)
        itm_logit3 = self.quality3(vl_embeddings)
        itm_logit3 = itm_logit3[:, :, 1].mean(dim=1)

        
        return itm_logit1, itm_logit2, itm_logit3
    

    def forward_llm(self, image, target_score1, target_score2, target_score3, tq_ids, tq_mask, ti_ids, ti_mask, to_ids, to_mask):

        self.query_tokens2 = self.zero_forward(self.query_tokens1) + self.query_tokens_initial




        target_scorer1 = target_score1.tolist()
        target_scorer2 = target_score2.tolist()
        target_scorer3 = target_score3.tolist()

        bs = image.size(0)
        # text_i = ["How would you rate the quality of this image? (Scale: 0-100, where 0 is low quality and 100 is high quality)"]
        # # text_i = ["what's the quality of the image?"]
        # text_instruct = []
        # for t in range(bs):
        #   text_instruct.extend(text_i)  
        
        
        # text_out = []
        # for t in range(bs):
        #     text_o = ["{:.2f}".format(target_scorer1[t]*100)]
        #     # text_o = ["The quality of the image is {} ".format(target_scorer1[t])]
        #     # text_o = ["The authenticity of the image is {} ".format(target_scorer2[t])]
        #     # text_o = ["The correspondence of the image is {} ".format(target_scorer3[t])]

        #     print(text_o)
        #     text_out.extend(text_o) 

        query_tokens = self.query_tokens2.expand(self.image_embeds.shape[0], -1, -1)
        # if self.qformer_text_input:
        # text_Qformer = self.tokenizer(
        #     text_instruct,
        #     padding='longest',
        #     truncation=True,
        #     max_length=self.max_txt_len,
        #     return_tensors="pt",
        # ).to(image.device)
        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(image.device)
        Qformer_atts = torch.cat([query_atts, tq_mask],dim=1)

        query_output = self.Qformer.bert(
            tq_ids,
            attention_mask=Qformer_atts,
            query_embeds=query_tokens,
            encoder_hidden_states=self.image_embeds,
            encoder_attention_mask=self.image_atts,
            return_dict=True,
        )
        # else:
        #     query_output = self.Qformer.bert(
        #         query_embeds=query_tokens,
        #         encoder_hidden_states=self.image_embeds,
        #         encoder_attention_mask=self.image_atts,
        #         return_dict=True,
        #     )

        inputs_llm = self.llm_proj(query_output.last_hidden_state[:,:query_tokens.size(1),:])
        atts_llm = torch.ones(inputs_llm.size()[:-1], dtype=torch.long).to(image.device)

        # self.llm_tokenizer.padding_side = "right"
        # self.llm_tokenizer.truncation_side = 'left'
        # text_input_tokens = self.llm_tokenizer(
        #     text_instruct,
        #     return_tensors="pt",
        #     padding="longest",
        #     truncation=True,
        #     max_length=self.max_txt_len,
        # ).to(image.device)
        

        # self.llm_tokenizer.truncation_side = 'right'
        # text_output_tokens = self.llm_tokenizer(
        #     [t + self.llm_tokenizer.eos_token for t in text_out],
        #     return_tensors="pt",
        #     padding="longest",
        #     truncation=True,
        #     max_length=self.max_output_txt_len,
        # ).to(image.device)

        llm_tokens, input_part_targets_len = self.concat_text_input_output(
            ti_ids,
            ti_mask,
            to_ids,
            to_mask,
        )

        # do not apply loss to the padding
        targets = llm_tokens['input_ids'].masked_fill(
            llm_tokens['input_ids'] == self.llm_tokenizer.pad_token_id, -100
        )

        # do not apply loss to the text input (i.e., instruction)
        for i, l in enumerate(input_part_targets_len):
            targets[i][:l] = -100

        # do not apply loss to the query tokens
        empty_targets = (
            torch.ones(atts_llm.size(), dtype=torch.long).to(image.device).fill_(-100)
        )
        targets = torch.cat([empty_targets, targets], dim=1)

        inputs_embeds = self.llm_model.get_input_embeddings()(llm_tokens['input_ids'])
        inputs_embeds = torch.cat([inputs_llm, inputs_embeds], dim=1)
        attention_mask = torch.cat([atts_llm, llm_tokens['attention_mask']], dim=1)

        with self.maybe_autocast():
            outputs = self.llm_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )

        loss = outputs.loss

        return loss


    @torch.no_grad()
    def generate(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=5,
        max_length=256,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.5,
        length_penalty=1,
        num_captions=1,
        temperature=1,
    ):
        self.llm_tokenizer.padding_side = "left"

        if "prompt" in samples.keys():
            prompt = samples["prompt"]
        else:
            prompt = self.prompt

        image = samples["image"]

        bs = image.size(0)

        if isinstance(prompt, str):
            prompt = [prompt] * bs
        else:
            assert len(prompt) == bs, "The number of prompts must be equal to the batch size."

        # For TextCaps
        if "ocr_tokens" in samples.keys() and "{}" in prompt[0]:
            prompt = [p.format(', '.join(samples['ocr_tokens'][i][:30])) for i, p in enumerate(prompt)]
        self.query_tokens2 = self.zero_forward(self.query_tokens1) + self.query_tokens_initial
        # self.query_tokens = self.zero_forward(self.query_tokens_initial)+self.query_tokens_in
        query_tokens = self.query_tokens2.expand(bs, -1, -1)
        if self.qformer_text_input:
            # remove ocr tokens in q_former (for eval textvqa)
            # qformer_prompt = prompt
            # qformer_prompt = ['Question: ' + qp.split(' Question: ')[1] for qp in qformer_prompt]

            text_Qformer = self.tokenizer(
                prompt,
                padding='longest',
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(image.device)
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(image.device)
            Qformer_atts = torch.cat([query_atts, text_Qformer.attention_mask], dim=1)

        # For video data
        if image.dim() == 5:
            inputs_llm, atts_llm = [], []
            for j in range(image.size(2)):
                this_frame = image[:,:,j,:,:]
                with self.maybe_autocast():
                    frame_embeds = self.ln_vision(self.visual_encoder(this_frame))
                frame_atts = torch.ones(frame_embeds.size()[:-1], dtype=torch.long).to(image.device)

                if self.qformer_text_input:
                    frame_query_output = self.Qformer.bert(
                        text_Qformer.input_ids,
                        attention_mask=Qformer_atts,
                        query_embeds=query_tokens,
                        encoder_hidden_states=frame_embeds,
                        encoder_attention_mask=frame_atts,
                        return_dict=True,
                    )
                else:
                    frame_query_output = self.Qformer.bert(
                        query_embeds=query_tokens,
                        encoder_hidden_states=frame_embeds,
                        encoder_attention_mask=frame_atts,
                        return_dict=True,
                    )
                frame_inputs_llm = self.llm_proj(frame_query_output.last_hidden_state[:,:query_tokens.size(1),:])
                frame_atts_llm = torch.ones(frame_inputs_llm.size()[:-1], dtype=torch.long).to(image.device)
                inputs_llm.append(frame_inputs_llm)
                atts_llm.append(frame_atts_llm)
            inputs_llm = torch.cat(inputs_llm, dim=1)
            atts_llm = torch.cat(atts_llm, dim=1)
        else:
            with self.maybe_autocast():
                image_embeds = self.ln_vision(self.visual_encoder(image))
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

            if self.qformer_text_input:
                query_output = self.Qformer.bert(
                    text_Qformer.input_ids,
                    attention_mask=Qformer_atts,
                    query_embeds=query_tokens,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_atts,
                    return_dict=True,
                )
            else:
                query_output = self.Qformer.bert(
                    query_embeds=query_tokens,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_atts,
                    return_dict=True,
                )

            inputs_llm = self.llm_proj(query_output.last_hidden_state[:,:query_tokens.size(1),:])
            atts_llm = torch.ones(inputs_llm.size()[:-1], dtype=torch.long).to(image.device)

        llm_tokens = self.llm_tokenizer(
            prompt,
            padding="longest",
            return_tensors="pt"
        ).to(image.device)

        with self.maybe_autocast():
            inputs_embeds = self.llm_model.get_input_embeddings()(llm_tokens.input_ids)
            inputs_embeds = torch.cat([inputs_llm, inputs_embeds], dim=1)
            attention_mask = torch.cat([atts_llm, llm_tokens.attention_mask], dim=1)

            outputs = self.llm_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                do_sample=use_nucleus_sampling,
                top_p=top_p,
                temperature=temperature,
                num_beams=num_beams,
                max_length=max_length,
                min_length=min_length,
                # eos_token_id=self.eos_token_id,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_return_sequences=num_captions,
            )

        outputs[outputs == 0] = 2 # convert output id 0 to 2 (eos_token_id)
        output_text = self.llm_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        output_text = [text.strip() for text in output_text]

        return output_text

    