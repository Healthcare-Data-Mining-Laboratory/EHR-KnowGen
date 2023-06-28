
import re
import torch
from torch import nn
from torch.nn import functional as F
import math
import torch.nn.utils.rnn as rnn_utils
from tqdm import tqdm
import numpy as np
import os
from collections import deque
import torch.optim as optim
import sys,logging
from transformers import AutoTokenizer, AutoModel
from transformers import T5Tokenizer
from models.ehrknowgen import EHRGeneration
tokenizer = T5Tokenizer.from_pretrained("t5-small")


class EHRKnowGen(nn.Module):
    def __init__(self,prompt_tokens,Mask_modality):
        super(EHRKnowGen, self).__init__()
        self.Mask_modality = Mask_modality
        self.text_encoder = EHRGeneration.from_pretrained("t5-small")
        n_tokens = prompt_tokens
        self.hidden_size = 512
        self.init_lab_prompt_value = self.text_encoder.encoder.embed_tokens.weight[:n_tokens].clone().detach()
        self.soft_lab_prompt = nn.Embedding(n_tokens, self.hidden_size)
        self.soft_lab_prompt.weight = nn.parameter.Parameter(self.init_lab_prompt_value)
        
        self.init_event_prompt_value = self.text_encoder.encoder.embed_tokens.weight[:n_tokens].clone().detach()
        self.soft_event_prompt = nn.Embedding(n_tokens, self.hidden_size)
        self.soft_event_prompt.weight = nn.parameter.Parameter(self.init_event_prompt_value)

        self.init_text_prompt_value = self.text_encoder.encoder.embed_tokens.weight[:n_tokens].clone().detach()
        self.soft_text_prompt = nn.Embedding(n_tokens, self.hidden_size)
        self.soft_text_prompt.weight = nn.parameter.Parameter(self.init_text_prompt_value)

        self.init_label_prompt_value = self.text_encoder.encoder.embed_tokens.weight[:n_tokens].clone().detach()
        self.soft_label_prompt = nn.Embedding(n_tokens, self.hidden_size)
        self.soft_label_prompt.weight = nn.parameter.Parameter(self.init_label_prompt_value)

        self.init_css_prompt_value = self.text_encoder.encoder.embed_tokens.weight[:n_tokens].clone().detach()
        self.soft_css_prompt = nn.Embedding(n_tokens, self.hidden_size)
        self.soft_css_prompt.weight = nn.parameter.Parameter(self.init_css_prompt_value)


        target_diagnosis_name_list =[
                'obstructive chronic bronchitis', 
                'acute respiratry failure', 
                'acute kidney failure', 
                'aortocoronary bypass', 
                'tobacco use', 
                'septicemia', 
                'septic shock', 
                'end stage renal', 
                'parox ventric tachycard', 
                'hypertensive chronic kidney',
                'invasive mechanical ventilation', 
                'endotracheal tube', 
                'entral infusion nutrit', 
                'venous cath', 
                'hemodialysis', 
                'renal dialysis', 
                'thoracentesis', 
                'small bowel endoscopy', 
                'percu abdominal drainage', 
                'mammary coronary artery bypass'
        ]
        self.label_ids = tokenizer(target_diagnosis_name_list, return_tensors="pt",padding=True)

    def forward(self, inputs_embeds,label_input,css_input,labels,Label_att):
        # output = self.text_encoder(input_ids=inputs_embeds,label_embeds = label_input,labels=labels, css_input = css_input, Mask_modality = self.Mask_modality, Label_att = Label_att)

        output = self.text_encoder(inputs_embeds=inputs_embeds,label_embeds = label_input,labels=labels, css_input = css_input, Mask_modality = self.Mask_modality, Label_att = Label_att)
        # output = self.text_encoder(inputs_embeds=[inputs_embeds,label_input,css_input,self.Mask_modality,Label_att],labels=labels)

        loss = output.loss
        css_matrix = output.css_matrix
        cross_att_score = output.cross_attentions
        # print()
        return loss,css_matrix,cross_att_score




       

    


   