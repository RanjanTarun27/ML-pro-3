# ML-pro-3
###################################################################
🦙 InstructLLaMA 3.2B: Lightweight Instruction Fine-Tuning with QLoRA

Fine-tuning Meta's LLaMA 3.2B model using QLoRA and PEFT (LoRA) for efficient and scalable instruction tuning. This project enables custom model adaptation with low GPU memory using Hugging Face's ecosystem.

---

## 🚀 Features

- ✅ Fine-tune LLaMA 3.2B on custom or open-source instruction datasets
- ✅ Memory-efficient 4-bit quantization using QLoRA
- ✅ Parameter-efficient fine-tuning (LoRA)
- ✅ Hugging Face Transformers + Datasets + PEFT support
- ✅ Google Colab & single GPU friendly

- !pip install -q transformers datasets accelerate peft bitsandbytes


python train.py \
  --model_name meta-llama/Meta-Llama-3-3B \
  --dataset_path ./data/alpaca.json \
  --output_dir ./llama3-3b-finetuned \
  --use_qlora \
  --use_lora \
  --lora_r 8 \
  --lora_alpha 16 \
  --num_train_epochs 3 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --bf16 \
  --logging_steps 20 \
  --save_strategy "epoch"


QLoRA is a quantization + LoRA technique that allows you to fine-tune large models like LLaMA 3 with just one GPU (12–16 GB) by:
1. Loading base model in 4-bit (using bitsandbytes)
2. Adding small, trainable LoRA adapters
3. Keeping base model weights frozen
   ################################################################################
   FOR EXAMPLE
   from transformers import AutoTokenizer, AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("llama3-3b-finetuned", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-3B")

prompt = "### Instruction:\nSummarize the Industrial Revolution in 3 points."
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))


💾 Model Weights & Export
You can upload your fine-tuned model to Hugging Face:
bash
Copy
Edit
from huggingface_hub import login
login()  # Paste your token
model.push_to_hub("your-username/llama3-3b-instruct-qlora")




