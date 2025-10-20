from tensorflow.keras.mixed_precision import experimental as mixed_precision

# Enable mixed precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)


import torch
from torch.cuda.amp import GradScaler, autocast

scaler = GradScaler()
for inputs, labels in dataloader:
    optimizer.zero_grad()
    with autocast():
        outputs = model(inputs)
        loss = criterion(outputs, labels)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

#bash
#pip install nvidia-dali

from nvidia.dali.pipeline import Pipeline
import nvidia.dali.fn as fn

class DALIPipeline(Pipeline):
    def __init__(self, batch_size, num_threads, device_id):
        super(DALIPipeline, self).__init__(batch_size, num_threads, device_id)
        self.input = fn.readers.file(file_root="data", random_shuffle=True)
        self.decode = fn.decoders.image(device="mixed")
        self.resize = fn.resize(self.decode, resize_x=224, resize_y=224)

    def define_graph(self):
        return self.resize

pipeline = DALIPipeline(batch_size=64, num_threads=4, device_id=0)
pipeline.build()

#bash
#pip install flash-attn

from flash_attn.flash_attention import FlashAttention

flash_attention_layer = FlashAttention()
output = flash_attention_layer(query, key, value)


#bash
#pip install transformers accelerate


from transformers import BertForSequenceClassification, Trainer, TrainingArguments

model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=4,
    fp16=True,  # Enable mixed precision
    learning_rate=5e-5,
    num_train_epochs=3,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

trainer.train()

#bash
#pip install onnxruntime-transformers

from onnxruntime.transformers import optimizer

optimized_model_path = optimizer.optimize_model(
    "bert-base-uncased.onnx",
    model_type="bert",
).save_model_to_file("optimized_bert.onnx")


#Optimizing ResNet-50 on RTX 4090

import torch.distributed as dist

dist.init_process_group("nccl")
model = torch.nn.parallel.DistributedDataParallel(model)
c. Bottleneck Analysis
Identify bottlenecks in training (e.g., data loading vs computation) using NVIDIA Nsight Systems or PyTorch Profiler.


from torch.profiler import profile, record_function, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    with record_function("model_inference"):
        model(inputs)

print(prof.key_averages().table(sort_by="cuda_time_total"))
