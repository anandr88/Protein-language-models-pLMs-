#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Input_csv:
test_data = pd.read_csv('train_data.csv')


from transformers import BertForMaskedLM, BertTokenizer, pipeline
from transformers import BertModel

import time

tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False )
model = BertModel.from_pretrained("Rostlab/prot_bert")
tokenizer.tokenize('A E T C Z A O')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
model.to(device)
# print(test_data['Sequence'].iloc[1])
# sequence_Example = ' '.join([*test_data['Sequence'].iloc[1]])
# encoded_input = tokenizer(sequence_Example, add_special_tokens=True, padding=True, is_split_into_words=True, return_tensors="pt").to(device)
# output = model(**encoded_input)
# output = output[1].detach().cpu().numpy()[0]
# output.shape, output

## Convert train_df protein sequence  into embeddings
prot_seq_input = test_data[['sequence']].drop_duplicates()
embeddings_list = []
for i in (range(0, len(prot_seq_input))):
    sequence_Example = ' '.join([*prot_seq_input['sequence'].iloc[i]])
    encoded_input = tokenizer(sequence_Example, add_special_tokens=True, padding=True, is_split_into_words=True, return_tensors="pt").to(device)
    output = model(**encoded_input)
    output = output[1].detach().cpu().numpy()[0]
    embeddings_list.append(output)
    if (i%100 == 0): print(i)
    if (i%500 == 0): time.sleep(3)
        
        
#get temp df of the embeddings:
temp = pd.DataFrame(embeddings_list)
#del embeddings_list
temp.columns = ['Feature_' + str(x) for x in temp.columns]
temp

prot_seq_input = pd.concat([prot_seq_input.reset_index(drop = True), temp.reset_index(drop = True)], axis = 1)
prot_seq_input.to_csv('prot_bert_feature.csv')

