import os, sys
sys.path.insert(0, os.path.abspath(".."))
import torch
from torch import nn
from torch.nn import functional as F
import math
import pandas as pd
import torch.nn.utils.rnn as rnn_utils
from sklearn.metrics import classification_report

from tqdm import tqdm
import numpy as np
import os
from collections import deque
import torch.optim as optim
from sklearn import metrics
from models.model_mimic_ehrknowgen import EHRKnowGen
# from clinical_bert import mllt
import copy
import json
from dataloader.dataloader_ehrknowgen import PatientDataset

SEED = 2019
torch.manual_seed(SEED)
import warnings
warnings.filterwarnings('ignore')
os.environ['CUDA_VISIBLE_DEVICES']="0"

from transformers import T5Tokenizer

## scigpu10  text_event_lab
## modify the transformer/generation/utils.py
tokenizer = T5Tokenizer.from_pretrained("t5-small")
class_3 = False
num_epochs = 10000
max_length = 10000
BATCH_SIZE = 3
prompt_tokens = 8
loss_ratio = [1,1]
Label_att = True
Mask_modality = False
Gen_Classfication = True
evaluation = True
pretrained = True
Freeze = False
SV_WEIGHTS = True
logs = True
visit = 'once'
Best_F1 = 0.28
date = "0610"

save_dir = "dir/"
save_name = f"name"
log_file_name = f'name.txt'

weight_dir = "xx.pth"

device1 = "cuda:0" if torch.cuda.is_available() else "cpu"
device1 = torch.device(device1)
device2 = "cuda:0" if torch.cuda.is_available() else "cpu"
device2 = torch.device(device2)
start_epoch =  0


if evaluation:
    visit = 'once'
    pretrained = True
    SV_WEIGHTS = False
    logs = False
 
    weight_dir = "xx.pth"

target_diagnosis_name_list = list(json.loads( open("label_ccs_icd_d.json",'r').read()).keys()) + list(json.loads( open("label_ccs_icd_p.json",'r').read()).keys())

css_name = [
   "diabetes mellitus with complications",
   "deficiency and other anemia",
   "complications of surgical procedures or medical care",
   "other or procedures on vessels other than head and neck",
   "other therapeutic procedures",
]


def clip_text(batch_size,max_length,vec,device):
    input_ids = vec['input_ids']
    attention_mask = vec['attention_mask']
    seq_ids = input_ids[:,[-1]]
    seq_mask = attention_mask[:,[-1]]
    input_ids_cliped = input_ids[:,:max_length-1]
    attention_mask_cliped = attention_mask[:,:max_length-1]
    input_ids_cliped = torch.cat([input_ids_cliped,seq_ids],dim=-1)
    attention_mask_cliped = torch.cat([attention_mask_cliped,seq_mask],dim=-1)
    vec = {'input_ids': input_ids_cliped,
    'attention_mask': attention_mask_cliped}
    return vec

def padding_text(batch_size,max_length,vec,device):
    input_ids = vec['input_ids']
    attention_mask = vec['attention_mask']
    sentence_difference = max_length - input_ids.shape[1]
    padding_ids = torch.ones((batch_size,sentence_difference), dtype = torch.long ).to(device)
    padding_mask = torch.zeros((batch_size,sentence_difference), dtype = torch.long).to(device)
    input_ids_padded = torch.cat([input_ids,padding_ids],dim=-1)
    attention_mask_padded = torch.cat([attention_mask,padding_mask],dim=-1)

    vec = {'input_ids': input_ids_padded,
    'attention_mask': attention_mask_padded}
    return vec
def cat_feature(max_length,text,event_list,lab_list):
    # text_input_ids = text['input_ids'][:,:max_length-1]
    # text_attention_mask = text['attention_mask'][:,:max_length-1]
    text_input_ids = text['input_ids'][:,:-1]
    text_attention_mask = text['attention_mask'][:,:-1]

    e_input_ids = event_list['input_ids'][:,1:]
    e_attention_mask = event_list['attention_mask'][:,1:]

    l_input_ids = lab_list['input_ids'][:,1:]
    l_attention_mask = lab_list['attention_mask'][:,1:]

    fuse_input_ids = torch.cat((text_input_ids,e_input_ids,l_input_ids),axis = -1)
    fuse_attention_mask  = torch.cat((text_attention_mask ,e_attention_mask,l_attention_mask),axis = -1)

    return {'input_ids':fuse_input_ids,
        'attention_mask': fuse_attention_mask}


def _cat_learned_embedding_to_input(model,fuse_input_ids,label_ids,css_ids,length_list,feature= "fuse") -> torch.Tensor:
    if feature == "fuse":

        fuse_inputs_embeds = model.text_encoder.encoder.embed_tokens(fuse_input_ids)
        n_batches = fuse_inputs_embeds.shape[0]
        learned_embeds_text = model.soft_text_prompt.weight.unsqueeze(0)
        learned_embeds_event = model.soft_event_prompt.weight.unsqueeze(0)
        learned_embeds_lab = model.soft_lab_prompt.weight.unsqueeze(0)
        inputs_embeds = [] 
        for b in range(n_batches):
            text_length,event_length = length_list[b][0],length_list[b][1]
            text = fuse_inputs_embeds[b,:text_length,:].unsqueeze(0)
            event = fuse_inputs_embeds[b,text_length:text_length+event_length,:].unsqueeze(0)
            lab = fuse_inputs_embeds[b,text_length+event_length:,:].unsqueeze(0)
            inputs_embeds.append(torch.cat((learned_embeds_text,text,learned_embeds_event,event,learned_embeds_lab,lab), dim=1))
        inputs_embeds = torch.cat(inputs_embeds,0)
        return inputs_embeds
    elif feature == "label":

        learned_embeds_label = model.soft_label_prompt.weight.unsqueeze(0).repeat(fuse_input_ids.shape[0],1,1)
        label_embeds =  model.text_encoder.encoder(input_ids = label_ids["input_ids"] , attention_mask = label_ids["attention_mask"])[0].mean(1).unsqueeze(0).repeat(fuse_input_ids.shape[0],1,1)
        inputs_embeds =  torch.cat((learned_embeds_label,label_embeds), dim=1)
        return inputs_embeds
    else:
        learned_embeds_css = model.soft_css_prompt.weight.unsqueeze(0).repeat(fuse_input_ids.shape[0],1,1)
        css_embeds =  model.text_encoder.encoder(input_ids = css_ids["input_ids"] , attention_mask = css_ids["attention_mask"])[0].mean(1).unsqueeze(0).repeat(fuse_input_ids.shape[0],1,1)
        inputs_embeds =  torch.cat((learned_embeds_css,css_embeds), dim=1)
        return inputs_embeds
def _extend_attention_mask(n_tokens,fuse_attention_mask,length_list,feature = "fuse"):
    n_batches = fuse_attention_mask.shape[0]
    attention_masks = []
    for b in range(n_batches):
        text_length,event_length = length_list[b][0],length_list[b][1]
        text = fuse_attention_mask[b,:text_length].unsqueeze(0)
        event = fuse_attention_mask[b,text_length:text_length+event_length].unsqueeze(0)
        lab = fuse_attention_mask[b,text_length+event_length:].unsqueeze(0)
        attention_masks.append(torch.cat((torch.full((1, n_tokens), 1).to(fuse_attention_mask.device),text,torch.full((1, n_tokens), 1).to(fuse_attention_mask.device),event,torch.full((1, n_tokens), 1).to(fuse_attention_mask.device),lab), dim=1))

    return torch.cat(attention_masks,0)

def collate_fn(data):    
    fuse_list = [d[0]for d in data]
    # event_list = [d[1] for d in data]
    # lab_list = [d[2] for d in data]
    length_list = [d[1] for d in data]
    label_list = [d[2] for d in data]
    label_name_list = [d[3] for d in data]
    label_css_parent = [d[4] for d in data]

    return fuse_list,length_list,label_list,label_name_list,label_css_parent



def fit(epoch,model,dataloader,loss_p,optimizer,flag='train'):
    global Best_F1
    
    if flag == 'train':
        device = device1
        model.train()

    else:
        device = device2
        model.eval()
    model.to(device)
    batch_loss_list = []

    y_list = []
    pred_list_f1 = []
    pred_list_roc = []
    all_modality_embedding = []
    all_embedding = []
    all_embedding_label = []
    all_text_embedding = []
    all_event_embedding = []
    all_lab_embedding = []

    all_css_label = []
    label_ids = tokenizer(target_diagnosis_name_list, return_tensors="pt",padding=True).to(device)
    css_ids = tokenizer(css_name, return_tensors="pt",padding=True,max_length = max_length).to(device)

    for i,(fuse_list,length_list,label_list,label_name_list,label_css_parent) in enumerate(tqdm(dataloader)):
        optimizer.zero_grad()
        # if i == 3: break
        if flag == "train":
     
            with torch.set_grad_enabled(True):
                label = torch.tensor(label_list).to(torch.float32).squeeze(1).to(device)
                # print("label: ",label_name_list)
                label_css_parent = torch.tensor(label_css_parent).to(torch.float32).squeeze(1).to(device)

                label_name_list = tokenizer(label_name_list, return_tensors="pt",padding=True,max_length = max_length).to(device)

                fuse_input = tokenizer(fuse_list, return_tensors="pt",padding=True,max_length = max_length).to(device)
                new_fuse_input = {
                    'inputs_embeds': _cat_learned_embedding_to_input(model,fuse_input['input_ids'],label_ids,css_ids,length_list,'fuse'),
                    'attention_mask':  _extend_attention_mask(prompt_tokens,fuse_input['attention_mask'],length_list,'fuse').to(device)
                }  
                label_input = _cat_learned_embedding_to_input(model,fuse_input['input_ids'],label_ids,css_ids,length_list,'label')
                css_input = _cat_learned_embedding_to_input(model,fuse_input['input_ids'],label_ids,css_ids,length_list,'css')
                loss1,css_matrix,cross_att_score = model(new_fuse_input['inputs_embeds'],label_input,css_input,label_name_list['input_ids'],Label_att)
                # print(css_matrix.squeeze(),label_css_parent.squeeze())

                if not Mask_modality:
                    if Label_att:
                        loss2 = loss_p(css_matrix,label_css_parent)
                        
                        loss = loss_ratio[0]*loss1+loss_ratio[1]*loss2
                    else:
                        loss = loss1
                    # print(loss1,loss2)

                else:
                    loss = loss1


                # output_sequences = model.text_encoder.generate(
                #     inputs_embeds = new_fuse_input['inputs_embeds'],
                #     label_embeds = label_input,
                #     css_input = css_input,
                #     Mask_modality = Mask_modality, 
                #     Label_att = Label_att,
                #     max_length = 400,
                #     num_return_sequences = 1,
                #     do_sample=False,  # disable sampling to test if batching affects output
                # )

                # pred_labels = tokenizer.batch_decode(output_sequences, skip_special_tokens=True)
                # print("pred: ",pred_labels)
                # pred = []
                # for pred_label in pred_labels:
                #     s_pred = [0]*len(target_diagnosis_name_list)
                #     for i,d in enumerate(target_diagnosis_name_list):                         
                #         if d in pred_label:
                #             s_pred[i] = 1  
                #     pred.append(s_pred) 

                pred = np.array([[0]*48]*label.shape[0])   
                y = np.array(label.cpu().data.tolist())


                # print("disaese label: ",y)
                # print("disease pred: ",pred)


                # print("..............................")

                y_list.append(y)
                pred_list_f1.append(pred)
                loss.backward(retain_graph=True)
                optimizer.step()
                batch_loss_list.append( loss.cpu().data )  

        else:
            with torch.no_grad():
              
                label = torch.tensor(label_list).to(torch.float32).squeeze(1).to(device)
                # print("label: ",label_name_list)

                label_css_parent = torch.tensor(label_css_parent).to(torch.float32).squeeze(1).to(device)

                label_name_list = tokenizer(label_name_list, return_tensors="pt",padding=True,max_length = max_length).to(device)
                # print(fuse_list)
                fuse_input = tokenizer(fuse_list, return_tensors="pt",padding=True,max_length = max_length).to(device)
                new_fuse_input = {
                    'inputs_embeds': _cat_learned_embedding_to_input(model,fuse_input['input_ids'],label_ids,css_ids,length_list,'fuse'),
                    'attention_mask':  _extend_attention_mask(prompt_tokens,fuse_input['attention_mask'],length_list,'fuse').to(device)
                }  
                label_input = _cat_learned_embedding_to_input(model,fuse_input['input_ids'],label_ids,css_ids,length_list,'label')

                css_input = _cat_learned_embedding_to_input(model,fuse_input['input_ids'],label_ids,css_ids,length_list,'css')

                # loss1,css_matrix,encoder_hiden_state = model(fuse_input['input_ids'],label_input,css_input,label_name_list['input_ids'],Label_att)

                loss1,css_matrix,cross_att_score = model(new_fuse_input['inputs_embeds'],label_input,css_input,label_name_list['input_ids'],Label_att)

                print(cross_att_score)
                # css_array = label_css_parent.squeeze().cpu().data.numpy()
                # all_css_label.append()
                if not Mask_modality:
                    if Label_att:
                        loss2 = loss_p(css_matrix,label_css_parent)
                        loss = loss_ratio[0]*loss1+loss_ratio[1]*loss2
                    else:
                        loss = loss1

                else:
                    loss = loss1



                #  3 0.8
                output_sequences = model.text_encoder.generate(
                    inputs_embeds = new_fuse_input['inputs_embeds'],
                    label_embeds = label_input,
                    num_beams = 1,
                    css_input = css_input,
                    Mask_modality = Mask_modality, 
                    Label_att = Label_att,
                    max_length = 500,
                    temperature = 0.8,
                    num_return_sequences = 1,
                )

                pred_labels = tokenizer.batch_decode(output_sequences, skip_special_tokens=True)
                # print("pred: ",pred_labels)
                pred = []
                for pred_label in pred_labels:
                    s_pred = [0]*len(target_diagnosis_name_list)
                    for i,d in enumerate(target_diagnosis_name_list):  
                        # print(pred_label)                       
                        if d in pred_label:
                            s_pred[i] = 1  
                    pred.append(s_pred) 

                pred = np.array(pred)   
                # print(pred.shape)
                y = np.array(label.cpu().data.tolist())

                # print("disaese label: ",y)
                # print("disease pred: ",pred)

                # print("..............................")
                y_list.append(y)
                pred_list_f1.append(pred)
                batch_loss_list.append( loss.cpu().data )  

    if Gen_Classfication:
        # target_diagnosis_name_list
        y_list_all = np.vstack(y_list)
        pred_list_all = np.vstack(pred_list_f1)
        top_10p = []
        top_20p = []
        top_40p = []
        top_60p = []
        top_80p = []
        top_100p = []
        top_10r = []
        top_20r = []
        top_40r = []
        top_60r = []
        top_80r = []
        top_100r = []
        top_10f = []
        top_20f = []
        top_40f = []
        top_60f = []
        top_80f = []
        top_100f = []    
        for n in range(y_list_all.shape[-1]):
            y_list_s = y_list_all[:,n]
            pred_list_f1_s = pred_list_all[:,n]
            acc = metrics.accuracy_score(y_list_s,pred_list_f1_s)
            precision = metrics.precision_score(y_list_s,pred_list_f1_s)
            recall =  metrics.recall_score(y_list_s,pred_list_f1_s)
            f1 = metrics.f1_score(y_list_s,pred_list_f1_s)
            number = sum(y_list_s)
            if 0<number <=64:
                top_10p.append(precision)
                top_10r.append(recall)
                top_10f.append(f1)

            if 64<number <=128:
                top_20p.append(precision)
                top_20r.append(recall)
                top_20f.append(f1)

            elif 128 <  number <= 185:
                top_40p.append(precision)
                top_40r.append(recall)
                top_40f.append(f1)

            elif 185 <  number <= 277:
                top_60p.append(precision)
                top_60r.append(recall)
                top_60f.append(f1)

            elif 277 <  number <= 438:
                top_80p.append(precision) 
                top_80r.append(recall)        
                top_80f.append(f1)        

            elif 438 <  number <= 1201:
                top_100p.append(precision)
                top_100r.append(recall)          
                top_100f.append(f1) 
        
        print(f"10% precision: {np.mean(np.array(top_10p))} | 20% precision: {np.mean(np.array(top_20p))} | 40% precision: {np.mean(np.array(top_40p))} | 60% precision: {np.mean(np.array(top_60p))} | 80% precision: {np.mean(np.array(top_80p))} | 100% precision: {np.mean(np.array(top_100p))}")            
        print(f"10% recall: {np.mean(np.array(top_10r))} | 20% recall: {np.mean(np.array(top_20r))} | 40% recall: {np.mean(np.array(top_40r))} | 60% recall: {np.mean(np.array(top_60r))} | 80% recall: {np.mean(np.array(top_80r))} | 100% recall: {np.mean(np.array(top_100r))}")            
        print(f"10% f1: {np.mean(np.array(top_10f))} | 20% f1: {np.mean(np.array(top_20f))}| 40% f1: {np.mean(np.array(top_40f))} | 60% f1: {np.mean(np.array(top_60f))} | 80% f1: {np.mean(np.array(top_80f))} | 100% f1: {np.mean(np.array(top_100f))}")            
    
    

        y_list = np.vstack(y_list)
        pred_list_f1 = np.vstack(pred_list_f1)
        # print(classification_report(y_list, pred_list_f1,target_names=target_diagnosis_name_list))
        acc = metrics.accuracy_score(y_list,pred_list_f1)

        precision_micro = metrics.precision_score(y_list,pred_list_f1,average='micro')
        recall_micro =  metrics.recall_score(y_list,pred_list_f1,average='micro')
        precision_macro = metrics.precision_score(y_list,pred_list_f1,average='macro')
        recall_macro =  metrics.recall_score(y_list,pred_list_f1,average='macro')

        f1_micro = metrics.f1_score(y_list,pred_list_f1,average="micro")
        f1_macro = metrics.f1_score(y_list,pred_list_f1,average="macro")
        total_loss = sum(batch_loss_list) / len(batch_loss_list)
    
    
        print("PHASE: {} EPOCH : {} | Micro Precision : {} | Macro Precision : {} | Micro Recall : {} | Macro Recall : {} | Micro F1 : {} |  Macro F1 : {} | ACC: {} | Total LOSS  : {}  ".format(flag,epoch + 1, precision_micro,precision_macro,recall_micro,recall_macro, f1_micro,f1_macro,acc,total_loss))
        if flag == 'test':
            if logs:
                with open(f'{log_file_name}', 'a+') as log_file:
                    log_file.write("PHASE: {} EPOCH : {} | Micro Precision : {} | Macro Precision : {} | Micro Recall : {} | Macro Recall : {} | Micro F1 : {} |  Macro F1 : {} |  ACC: {} | Total LOSS  : {}  ".format(flag,epoch + 1, precision_micro,precision_macro,recall_micro,recall_macro, f1_micro,f1_macro,acc,total_loss)+'\n')
                    log_file.close()
            if SV_WEIGHTS:
                if f1_micro > Best_F1:
                    Best_F1 = f1_micro
                    PATH=f"{save_dir}/{save_name}_epoch_{epoch}_loss_{round(float(loss),4)}_f1_micro_{round(float(f1_micro),4)}_f1_macro_{round(float(f1_macro),4)}.pth"
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(best_model_wts, PATH)
    else:
        total_loss = sum(batch_loss_list) / len(batch_loss_list)
        print("PHASE: {} EPOCH : {} | Loss: {}".format(flag,epoch + 1,total_loss))
        if flag == 'test':
            if SV_WEIGHTS:
                if total_loss <= 0.2 :
                    PATH=f"{save_dir}/{save_name}_epoch_{epoch}_loss_{round(float(loss),4)}.pth"
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(best_model_wts, PATH)



if __name__ == '__main__':


    train_dataset = PatientDataset(f"dir/", class_3 = class_3, visit = visit,Mask_modality=Mask_modality,Gen_Classfication = Gen_Classfication,flag="train")
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn,shuffle = True,drop_last = True)
    test_dataset = PatientDataset(f"dir/",class_3 = class_3, visit = visit,Mask_modality=Mask_modality,Gen_Classfication = Gen_Classfication, flag="test")
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn,shuffle = True,drop_last = True)

    train_length = train_dataset.__len__()
    test_length = test_dataset.__len__()

    print(train_length)
    print(test_length)

    model = EHRKnowGen(prompt_tokens,Mask_modality)
    loss_p = nn.BCELoss()

    if pretrained:
        print(f"loading weights: {weight_dir}")
        pretrained_weight = torch.load(weight_dir,map_location=torch.device(device2))
        # # if you need to change prompt length
        # del pretrained_weight["soft_lab_prompt.weight"]
        # del pretrained_weight["soft_event_prompt.weight"]
        # del pretrained_weight["soft_css_prompt.weight"]
        # del pretrained_weight["soft_label_prompt.weight"]
        # del pretrained_weight["soft_text_prompt.weight"]
        model.load_state_dict(pretrained_weight, strict=False)

    ### freeze parameters ####
    optimizer = optim.Adam(model.parameters(True), lr = 5e-5)

    if Freeze:
        for (i,child) in enumerate(model.children()):
            if i == 0:
                for param in child.parameters():
                    param.requires_grad = False
    ##########################
    if evaluation:
        fit(1,model,testloader,loss_p,optimizer,flag='test')
    else:
        for epoch in range(start_epoch,num_epochs):

            fit(epoch,model,trainloader,loss_p,optimizer,flag='train')
            fit(epoch,model,testloader,loss_p,optimizer,flag='test')










