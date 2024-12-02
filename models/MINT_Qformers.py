import logging
import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn
from models.blip2r import Blip2Base, disabled_train
from config.options_infer import *
class MINTqformer1(Blip2Base):

    def __init__(
        self,
        vit_model="eva_clip_g",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp32",
        freeze_vit=False,
        num_query_token=32,
        embed_dim=256,
        max_txt_len=128,
        max_output_txt_len=256,
        qformer_text_input=True,
    ):
        super().__init__()      
        
        self.tokenizer = self.init_tokenizer(truncation_side="left")
        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )

        self.image_layer_num = 38
        # image_fix_num = "blocks.{}".format(int(self.image_layer_num * opts.fix_rate))
    
        # for name, parms in self.visual_encoder.named_parameters():
        #     parms.requires_grad_(False)
        #     if image_fix_num in name:
        #         break

        freeze_vit = False
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            logging.info("freeze vision encoder")

        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features
        )
        self.Qformer.resize_token_embeddings(len(self.tokenizer))

        state_dict = self.Qformer.state_dict()
        for name, param in self.Qformer.named_parameters():
            if "_query" in name:
                key_orig = name.replace("_query", "")
                param.data.copy_(state_dict[key_orig])

        self.vision_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)
        self.text_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)

        self.quality1 = self.quality_regression(self.Qformer.config.hidden_size, 48, 3)
        self.quality2 = self.quality_regression(self.Qformer.config.hidden_size, 48, 3)
        self.quality3 = self.quality_regression(self.Qformer.config.hidden_size, 48, 3)
        self.temp = nn.Parameter(0.07 * torch.ones([]))

        self.max_txt_len = max_txt_len
        self.Qformer.cls = None
    
        for name, param in self.Qformer.named_parameters():
            param.requires_grad = False

        self.max_txt_len = max_txt_len
        self.max_output_txt_len = max_output_txt_len
        self._lemmatizer = None

        self.qformer_text_input = qformer_text_input
       


    def quality_regression(self,in_channels, middle_channels, out_channels):
        regression_block = nn.Sequential(
            nn.Linear(in_channels, middle_channels),
            nn.Linear(middle_channels, out_channels),          
        )

        return regression_block

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

        self.image_embeds = self.ln_vision(self.visual_encoder(image))
        self.image_atts = torch.ones(self.image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )
        bs = image.size(0)

        query_tokens_itm = self.query_tokens.expand(text_ids.shape[0], -1, -1)
        query_atts_itm = torch.ones(query_tokens_itm.size()[:-1], dtype=torch.long).to(
            image.device
        )
        attention_mask_all = torch.cat([query_atts_itm, text_mask], dim=1)

        output_itm = self.Qformer.bert(
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

        
        return itm_logit1, itm_logit2, itm_logit3, vl_embeddings
    
class MINTqformer2(Blip2Base):
    def __init__(
        self,
        vit_model="eva_clip_g",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp32",
        freeze_vit=False,
        num_query_token=32,
        embed_dim=256,
        llm_model="vicuna7b",
        prompt="",
        max_txt_len=128,
        max_output_txt_len=256,
        qformer_text_input=True,
    ):
        super().__init__()      
        from transformers import LlamaTokenizer
        from models.modeling_llama import LlamaForCausalLM
        
        self.tokenizer = self.init_tokenizer(truncation_side="left")
        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )

        self.image_layer_num = 38
        # image_fix_num = "blocks.{}".format(int(self.image_layer_num * opts.fix_rate))
    
        # for name, parms in self.visual_encoder.named_parameters():
        #     parms.requires_grad_(False)
        #     if image_fix_num in name:
        #         break

        freeze_vit = True
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            logging.info("freeze vision encoder")

        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features
        )
        
        self.Qformer.resize_token_embeddings(len(self.tokenizer))

        state_dict = self.Qformer.state_dict()
        for name, param in self.Qformer.named_parameters():
            if "_query" in name:
                key_orig = name.replace("_query", "")
                param.data.copy_(state_dict[key_orig])

        self.vision_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)
        self.text_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)

        self.temp = nn.Parameter(0.07 * torch.ones([]))

        self.max_txt_len = max_txt_len
        self.Qformer.cls = None
        llm_model = opts.llm
        self.llm_tokenizer = LlamaTokenizer.from_pretrained(llm_model, use_fast=False, truncation_side="left")
        self.llm_model = LlamaForCausalLM.from_pretrained(
            llm_model, torch_dtype=torch.float16
        )
        self.llm_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.llm_tokenizer.add_special_tokens({'bos_token': '</s>'})
        self.llm_tokenizer.add_special_tokens({'eos_token': '</s>'})
        self.llm_tokenizer.add_special_tokens({'unk_token': '</s>'})

        self.llm_model.resize_token_embeddings(len(self.llm_tokenizer))

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
        return self.zero_module(nn.Conv1d(channels, channels, 1, padding=0))
    

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
    

    def forward_llm(self, vl_embeddings, image, tq_ids, tq_mask, ti_ids, ti_mask, to_ids, to_mask):
        
        self.image_embeds = self.ln_vision(self.visual_encoder(image))
        self.image_atts = torch.ones(self.image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )
        query_tokens2 = self.zero_forward(vl_embeddings) + self.query_tokens

        bs = image.size(0)

        query_tokens = query_tokens2.expand(self.image_embeds.shape[0], -1, -1)
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
        back_vector = query_output.last_hidden_state[:,:query_tokens.size(1),:]

        inputs_llm = self.llm_proj(query_output.last_hidden_state[:,:query_tokens.size(1),:])
        atts_llm = torch.ones(inputs_llm.size()[:-1], dtype=torch.long).to(image.device)
        

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

        return loss, back_vector
    
    @torch.no_grad()
    def generate(
        self,
        vl_embeddings,
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
        query_tokens2 = self.zero_forward(vl_embeddings) + self.query_tokens
        query_tokens = query_tokens2.expand(bs, -1, -1)
        if self.qformer_text_input:
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
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_return_sequences=num_captions,
            )

        outputs[outputs == 0] = 2 
        output_text = self.llm_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        output_text = [text.strip() for text in output_text]

        return output_text

    