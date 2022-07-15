from transformers import AutoTokenizer
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM
from transformers import Seq2SeqTrainingArguments
from transformers import DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainer
from datasets import load_metric

rouge_score = load_metric("rouge")
model_checkpoint = "google/mt5-small"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

train_dataset = load_dataset("csv", data_files="train.csv")

train_size = 15
test_size = 5

train_val = train_dataset["train"].train_test_split(
    train_size=train_size, test_size=test_size, seed=42
)

print(train_val)
max_input_length = 512
max_target_length = 30


def preprocess_function(examples):
    model_inputs = tokenizer(
        examples["data"], max_length=max_input_length, padding=True, truncation=True
    )
    # 标签处理方法人家也提供了
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            examples["label"], max_length=max_target_length, padding=True, truncation=True
        )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


tokenized_datasets = train_val.map(preprocess_function, batched=True)
print(tokenized_datasets)
# print(tokenized_datasets['train']['input_ids'])

model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
batch_size = 1
num_train_epochs = 1000
# 每一个epoch打印
logging_steps = len(tokenized_datasets["train"]) // batch_size
model_name = model_checkpoint.split("/")[-1]


def compute_metrics(eval_pred):
    predictions, labels = eval_pred  # 没有考虑特殊字符，实际用需根据你的tokenizer筛选剔除这些
    result = rouge_score.compute(
        predictions=predictions, references=labels, use_stemmer=True
    )
    # 返回结果
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    return {k: round(v, 4) for k, v in result.items()}


tokenized_datasets = tokenized_datasets.remove_columns(
    train_dataset["train"].column_names
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

args = Seq2SeqTrainingArguments(
    output_dir=f"{model_name}-finetuned-amazon-en-es",
    evaluation_strategy="epoch",
    learning_rate=5.6e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=1,
    num_train_epochs=num_train_epochs,
    predict_with_generate=True,  # 评估的时候需要生成的结果
    logging_steps=logging_steps,
    save_strategy='steps',
    save_steps=2000,
)

trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()
