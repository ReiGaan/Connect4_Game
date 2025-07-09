import torch
from torch.profiler import profile, record_function, ProfilerActivity
from agents.alphazero.network import Connect4Net

# Initialize model and dummy input
model = Connect4Net()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Adjust the input size based on the model's requirements
dummy_input = torch.randn(1, 3, 6, 7).to(device)  # Batch size 1, 3 channels, 6x7 board

# GPU profiling
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./gpu_profile_logs'),
    record_shapes=True,
    with_stack=True
) as prof:
    with record_function("model_inference"):
        output = model(dummy_input)

# Write profiling results to a file
with open('gpu_profile_results.log', 'w') as log_file:
    log_file.write(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))