import torch
from PIL import Image
from transformers import BertTokenizer
from models.MINTIQA import mintiqa
from config.options_infer import *

def init_tokenizer():
    
    tokenizer = BertTokenizer.from_pretrained(opts.tokenizer)
    #tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenizer.add_special_tokens({'bos_token':'[DEC]'})
    tokenizer.add_special_tokens({'additional_special_tokens':['[ENC]']})       
    tokenizer.enc_token_id = tokenizer.additional_special_tokens_ids[0]  
    return tokenizer

if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    #Choose from the following questions
    text_instruct = [
                     "How is the quality of the image?",
                     "How is the authenticity of the image?",
                     "How is the correspondence of the image?",
                     "Assess the image from three perspectives: quality, authenticity, and correspondence"
                      ]
    Question = text_instruct[3]
    
    #Set the image path
    img_path ='0.png'
    #Set the corresponding prompt to the image
    prompt = 'a corgi'
    
    
    model_path = 'ckpt/MINTIQA.pt'
    state_dict = torch.load(model_path, map_location='cpu')
    model = mintiqa(device).to(device)
    msg = model.load_state_dict(state_dict,strict=False)
    print(msg)
    print("checkpoint loaded")
    tokenizer = init_tokenizer()


    
    with torch.no_grad():  
        pil_image = Image.open(img_path)
        image = model.preprocess(pil_image)
        image = image.unsqueeze(0)   
        text_input = tokenizer(prompt, padding='max_length', truncation=True, max_length=35, return_tensors="pt")
        text_ids = text_input.input_ids
        text_mask = text_input.attention_mask
        text_ids = text_ids.view(text_ids.shape[0], -1).to(device) # [batch_size, seq_len]
        text_mask = text_mask.view(text_mask.shape[0], -1).to(device)
        img = image.to(device)
        emb_better1, emb_better2, emb_better3, quality_vector = model.qformer1.forward_3score(image = img, text_ids=text_ids, text_mask=text_mask)
        Answer =  model.qformer2.generate(quality_vector, samples = {"image": img, "prompt": Question})[0].split('<s>')[1] 
        quality_score  = emb_better1.detach().cpu().numpy().item() * 100
        authenticity_score  = emb_better2.detach().cpu().numpy().item() * 100
        correspondence_score = emb_better3.detach().cpu().numpy().item() * 100
        
        print('The quality score is:',quality_score,'\n')
        print('The authenticity score is:',authenticity_score,'\n')
        print('The correspondence score is:',correspondence_score,'\n')
        print('Answer:', Answer)
    # save_model_llm(model)