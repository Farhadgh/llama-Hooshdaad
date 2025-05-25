# finetune_llama_vast.py

from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling, pipeline
from datasets import load_dataset, Dataset
import torch
import os
import json

# Load environment token (you must set HF_TOKEN in Vast.ai instance environment)
hf_token = os.getenv("HF_TOKEN")

# Load model and tokenizer
model_name = "meta-llama/Llama-4-Scout-17B-16E-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
    use_auth_token=hf_token
)

# Load your dataset from JSON file
with open("data.json", "r", encoding="utf-8") as f:
    raw_data = json.load(f)

# Transform dataset to list of dicts with 'text' key
data = []
for item in raw_data:
    for s in item["scenarios"]:
        prompt = f"سوال: {s['Question']}\nپاسخ: {s['Answer']}\n"
        context = f"متن قانون: {item['text']}\nمبانی قانونی: {item.get('legal_reasoning', '')}"
        full_text = f"{prompt}{context}"
        data.append({"text": full_text})

# Convert to HuggingFace dataset
dataset = Dataset.from_list(data)

def tokenize(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=1024)

dataset = dataset.map(tokenize, batched=True)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir="./llama-hooshdad-finetuned",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-5,
    fp16=True,
    logging_dir="./logs",
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=10,
    push_to_hub=True,
    hub_model_id="your-username/llama-hooshdad",
    hub_token=hf_token
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

trainer.train()

# Push final model
trainer.push_to_hub()

# ✅ Simple test pipeline after training
print("\n⏳ تست مدل پس از فاین‌تیون:")
test_pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")
test_questions = [
    "نظام مهندسی و کنترل ساختمان چیست؟",
    "آیا شرایط معامله در تعیین میزان غبن موثر است؟",
    "برات به چند نوع وعده می‌تواند باشد؟",
    "آیا ولی دم می تواند قبل از فوت مجنی‌علیه مرتکب را قصاص کند؟",
    "ماده ۳۳۱ قانون آیین دادرسی مدنی چه کاربردی دارد؟"
]

for q in test_questions:
    prompt = f"سوال: {q}\nپاسخ:"
    outputs = test_pipe(prompt, max_new_tokens=200, do_sample=True, top_p=0.95, temperature=0.7)
    print(f"\n🔹 سوال: {q}\n🔸 پاسخ مدل: {outputs[0]['generated_text'].replace(prompt, '').strip()}")

