import json
import matplotlib.pyplot as plt

model_name = "whisper_large_v3"
# Path to your JSON file
json_path = "/shares/rndsounds/wake_up_word/genAI/ac/stt/Verbit/large_v3/checkpoint-16000/trainer_state.json"
# json_path = "/shares/rndsounds/wake_up_word/genAI/ac/stt/Verbit/large_v3_turbo/checkpoint-18000/trainer_state.json"



# Read the JSON data
with open(json_path, 'r') as f:
    data = json.load(f)

# Extract step-wise log history
log_history = data.get('log_history', [])

# Separate metrics
train_steps, train_loss = [], []
eval_steps, eval_loss, eval_wer = [], [], []

for entry in log_history:
    step = entry.get("step")
    if "loss" in entry:
        train_steps.append(step)
        train_loss.append(entry["loss"])
    if "eval_loss" in entry:
        eval_steps.append(step)
        eval_loss.append(entry["eval_loss"])
    if "eval_wer" in entry:
        eval_wer.append(entry["eval_wer"])

# Ensure alignment of steps for eval_wer
eval_wer_steps = eval_steps[:len(eval_wer)]

# Plotting
plt.figure(figsize=(12, 6))

# Loss Plot
plt.subplot(1, 2, 1)
plt.plot(train_steps, train_loss, label='Train Loss', marker='o')
plt.plot(eval_steps, eval_loss, label='Eval Loss', marker='x')
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Train vs Eval Loss per Step")
plt.legend()
plt.grid(True)

# Eval WER Plot
plt.subplot(1, 2, 2)
plt.plot(eval_wer_steps, eval_wer, label='Eval WER', color='purple', marker='s')
plt.xlabel("Step")
plt.ylabel("WER")
plt.title("Eval Word Error Rate (WER)")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()