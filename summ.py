# from transformers import BartTokenizer, BartForConditionalGeneration, Trainer, TrainingArguments

# metric_rouge = load_metric("rouge")
# metric_bleu = load_metric("sacrebleu")
# metric_bert = load_metric("bertscore")
# metric_meteor = load_metric("meteor")

# def compute_metrics(eval_pred):
#     predictions, labels = eval_pred
#     predictions

from datasets import Dataset, load_metric
import json, io
from transformers import (
    AutoTokenizer, 
    DataCollatorForSeq2Seq, 
    AutoModelForSeq2SeqLM, 
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer,
    T5ForConditionalGeneration,
    T5Tokenizer)
import evaluate
import numpy as np

with io.open(fr'C:\Users\MRZholus\Desktop\python_test\marked_dataset_f.json', 'r', encoding='utf-8') as file:
    full_data = json.load(file)
    data = [{"text": item["text"], "summary": item["summary"]} for item in full_data]
    dataset = Dataset.from_list(data)
    dataset = dataset.train_test_split(test_size=0.2)
    
checkpoint = "google/mt5-base"
tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=False)

prefix = "summarize: "

def preprocess_function(examples):
    inputs = [prefix + doc for doc in examples["text"]]
    model_inputs = tokenizer(inputs, max_length=1024, truncation=True)
    labels = tokenizer(text_target=examples["summary"], max_length=128, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_dataset = dataset.map(preprocess_function, batched=True)
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=checkpoint)

rouge = evaluate.load("rouge")
bertscore = evaluate.load("bertscore")
bleu = evaluate.load("sacrebleu")
# meteor = evaluate.load("meteor")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    res_rouge = rouge.compute(predictions=decoded_preds, references=decoded_labels)
    res_bleu = bleu.compute(predictions=decoded_preds, references=decoded_labels)
    res_bert = bertscore.compute(predictions=decoded_preds, references=decoded_labels)
    
    # prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    # res_rouge["gen_len"] = np.mean(prediction_lens)
    # return {k: round(v, 4) for k, v in res_rouge.items()}
    return {
        "rouge": res_rouge,
        "bleu": res_bleu,
        "bert": res_bert
    }

model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

training_args = Seq2SeqTrainingArguments(
    output_dir="my_awesome_summ_model",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=10,
    predict_with_generate=True,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.evaluate()