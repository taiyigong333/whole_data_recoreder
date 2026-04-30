from transformers import AutoModel, AutoProcessor

model = AutoModel.from_pretrained("/root/autodl-tmp/X-VLA-Pt", trust_remote_code=True)
processor = AutoProcessor.from_pretrained("/root/autodl-tmp/X-VLA-Pt", trust_remote_code=True)

print("Starting X-VLA-Pt server...")
model.run(processor, host="0.0.0.0", port=8000)
