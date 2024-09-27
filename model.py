from transformers import BitsAndBytesConfig, LlavaNextVideoForConditionalGeneration, LlavaNextVideoProcessor
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

processor = LlavaNextVideoProcessor.from_pretrained("llava-hf/LLaVA-NeXT-Video-7B-hf", cache_dir='D:\Huggingface\caches')

model = LlavaNextVideoForConditionalGeneration.from_pretrained(
    "llava-hf/LLaVA-NeXT-Video-7B-hf",
    quantization_config=quantization_config,
    cache_dir='D:\Huggingface\caches'
)

model.to(device)