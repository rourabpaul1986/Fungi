import itertools
import time
import warnings
#from peft import LoraConfig, get_peft_model
from transformers import BertForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from matplotlib import pyplot as plt
from datasets import load_dataset
import torch
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, matthews_corrcoef, roc_auc_score
#import huggingface_hub

# huggingface_hub.login(token=hf_token)

# Suppress warnings
warnings.filterwarnings("ignore")

# # Layer configurations
# attention_plus_feed_forward = [
#     "bert.encoder.layer.0.attention.self.query",
#     "bert.encoder.layer.0.attention.self.key",
#     "bert.encoder.layer.0.attention.self.value",
#     "bert.encoder.layer.0.attention.output.dense",
#     "bert.encoder.layer.0.intermediate.dense",
#     "bert.encoder.layer.0.output.dense",
#     "bert.encoder.layer.1.attention.self.query",
#     "bert.encoder.layer.1.attention.self.key",
#     "bert.encoder.layer.1.attention.self.value",
#     "bert.encoder.layer.1.attention.output.dense",
#     "bert.encoder.layer.1.intermediate.dense",
#     "bert.encoder.layer.1.output.dense"
# ]


tokenizer = AutoTokenizer.from_pretrained('/leonardo_work/IscrC_SEMITONE/DNABERT6')
# Function to preprocess the dataset
def preprocess_function(examples):
    try:
        return tokenizer(
            examples['sequence'],
            padding='max_length',
            truncation=True,
            max_length=512
        )
    except KeyError:
        return tokenizer(
            examples['Sequence'],
            padding='max_length',
            truncation=True,
            max_length=512
        )


def add_labels(examples):
    try:
        examples['labels'] = examples['label']
        return examples
    except KeyError:
        examples['labels'] = examples['Label']
        return examples

def create_task_dataset(task_name):
    if task_name == 'tfbs':
        return load_dataset('csv', data_files='/kaggle/working/tfbs.csv', split='train[0:10000]'), load_dataset('csv', data_files='/kaggle/working/tfbs.csv', split='train[10001:13122]')

    elif task_name == 'dnasplice':
        return load_dataset('csv', data_files='/kaggle/working/dnasplice.csv', split='train[0:10000]'), load_dataset('csv', data_files='/kaggle/working/dnasplice.csv', split='train[10001:13122]')

    elif task_name == 'dnaprom':
        return load_dataset('csv', data_files='/kaggle/working/dnaprom.csv', split='train[0:10000]'), load_dataset('csv', data_files='/kaggle/working/dnaprom.csv', split='train[10001:13122]')
    elif task_name == 'specie_prediction':
        return load_dataset('csv',data_files = '/leonardo_work/IscrC_SEMITONE/Fungi/Fungi_samples_kmer6.csv', split = 'train[0:70000]'), load_dataset('csv', data_files='/leonardo_work/IscrC_SEMITONE/Fungi/Fungi_samples_kmer6.csv', split='train[70001:99999]')
    else:
        raise ValueError(f"Unknown task: {task_name}")

def create_dataset_maps(train_dataset, test_dataset):
    train_dataset = train_dataset.map(preprocess_function, batched=True)
    train_dataset = train_dataset.map(add_labels)
    test_dataset = test_dataset.map(preprocess_function, batched=True)
    test_dataset = test_dataset.map(add_labels)
    return train_dataset, test_dataset

def train_model(train_dataset, test_dataset, model, task, model_name, config_name):
    def specificity_score(y_true, y_pred):
        true_negatives = np.sum((y_pred == 0) & (y_true == 0))
        false_positives = np.sum((y_pred == 1) & (y_true == 0))
        specificity = true_negatives / (true_negatives + false_positives + np.finfo(float).eps)
        return specificity

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        y_pred = logits[:, 1]

        accuracy = accuracy_score(labels, predictions)
        recall = recall_score(labels, predictions)
        specificity = specificity_score(labels, predictions)
        mcc = matthews_corrcoef(labels, predictions)
        roc_auc = roc_auc_score(labels, y_pred)
        precision = precision_score(labels, predictions)
        f1 = f1_score(labels, predictions)

        true_pos = np.sum((predictions == 1) & (labels == 1))
        true_neg = np.sum((predictions == 0) & (labels == 0))
        false_pos = np.sum((predictions == 1) & (labels == 0))
        false_neg = np.sum((predictions == 0) & (labels == 1))

        return {
            'accuracy': accuracy,
            'recall': recall,
            'specificity': specificity,
            'mcc': mcc,
            'roc_auc': roc_auc,
            'precision': precision,
            'f1': f1,
            'true_pos': true_pos,
            'true_neg': true_neg,
            'false_pos': false_pos,
            'false_neg': false_neg
        }

    # Define the training arguments
    training_arguments = TrainingArguments(
        output_dir=f"/leonardo_work/IscrC_SEMITONE/Fungi/outputs/{task}/{model_name}_{config_name}",
        num_train_epochs=25,
        fp16=False,
        bf16=False,
        per_device_train_batch_size=20,
        per_device_eval_batch_size=10,
        gradient_accumulation_steps=2,
        gradient_checkpointing=True,
        max_grad_norm=0.3,
        learning_rate=4e-4,
        weight_decay=0.01,
        optim="paged_adamw_32bit",
        lr_scheduler_type="linear",
        max_steps=-1,
        warmup_ratio=0.03,
        group_by_length=True,
        save_steps=1000,
        logging_steps=25,
        dataloader_pin_memory=False,
        report_to='tensorboard',
        gradient_checkpointing_kwargs={'use_reentrant': False}
    )

    trainer = Trainer(
        model=model,
        args=training_arguments,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    start_time = time.time()
    trainer.train()
    end_time = time.time()

    total_time = end_time - start_time
    metrics = trainer.evaluate()

    return total_time, metrics

# Task loop
# task_list = ['dnasplice', 'tfbs', 'dnaprom']
task_list = ['specie_prediction']
log_file = "/leonardo_work/IscrC_SEMITONE/Fungi/training_log.txt"
model_name = '/leonardo_work/IscrC_SEMITONE/DNABERT6'
for task in task_list:
    print(f"Running TASK : {task}")
    train_dataset, test_dataset = create_task_dataset(task)
    train_dataset, test_dataset = create_dataset_maps(train_dataset, test_dataset)
    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])


    # Train the base model first
    base_model = BertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2
    )
    config_name = "base_model"
    print(f"Training MODEL : {config_name} for task : {task}")
    training_time, metrics = train_model(train_dataset, test_dataset, base_model, task, model_name, config_name)
    with open(log_file, "a") as log:
        log.write(f"Task: {task}, Model: {model_name}, Config: {config_name}, Training Time: {training_time}, Metrics: {metrics}\n")
