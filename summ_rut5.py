import torch 
from transformers import T5ForConditionalGeneration, T5Tokenizer
from tqdm.auto import trange
import random
import numpy as np
import json, io


raw_model = 'cointegrated/rut5-base-multitask' 
model = T5ForConditionalGeneration.from_pretrained(raw_model)
tokenizer = T5Tokenizer.from_pretrained(raw_model)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)


batch_size = 16
report_steps = 50
epochs = 7

model.train()
losses = []

with io.open(fr'marked_dataset_f.json', 'r', encoding='utf-8') as file:
    full_data = json.load(file)
    # data = [{"text": item["text"], "summary": item["summary"]} for item in full_data]
    pairs = [[item["text"], item["summary"]] for item in full_data]

    
for epoch in range(epochs):
    print('EPOCH', epoch)
    random.shuffle(pairs)
    for i in trange(0, int(len(pairs) / batch_size)):
        batch = pairs[i * batch_size: (i + 1) * batch_size]

        x = tokenizer([p[0] for p in batch], return_tensors='pt', padding=True)
        y = tokenizer([p[1] for p in batch], return_tensors='pt', padding=True)

        y.input_ids[y.input_ids == 0] = -100

        loss = model(
            input_ids=x.input_ids,
            attention_mask=x.attention_mask,
            labels=y.input_ids,
            decoder_attention_mask=y.attention_mask,
            return_dict=True
        ).loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        losses.append(loss.item())
        if i % report_steps == 0:
            print('step', i, 'loss', np.mean(losses[-report_steps:]))
