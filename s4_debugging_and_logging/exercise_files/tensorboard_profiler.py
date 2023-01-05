import torch
import torchvision.models as models
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler

model = models.resnet34()
inputs = torch.randn(5, 3, 224, 224)

with profile(activities=[ProfilerActivity.CPU], record_shapes=True, profile_memory = True, on_trace_ready=tensorboard_trace_handler("./log/resnet34")) as prof:
   model(inputs)

print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

prof.export_chrome_trace("trace.json")