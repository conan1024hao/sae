from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from sae import PeftSaeModel


model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct", torch_dtype="auto", device_map="auto"
)
min_pixels = 256*28*28
max_pixels = 1280*28*28
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "/workspace/sae/images/panda_attacked.png",
            },
            {"type": "text", "text": "Describe this image."},
        ],
    }
]
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cuda")

generated_ids = model.generate(**inputs, max_new_tokens=128, do_sample=False)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print("============================================")
print("Original Model Output:")
print(output_text[0])

# "conan1024hao/qwen2.5-vl-3b-sae-lm-layer33-finevisionmax-500k"
# "conan1024hao/qwen2.5-vl-3b-sae-projection-mlp2-finevisionmax-500k"
# "conan1024hao/qwen2.5-vl-3b-sae-vision-block30-finevisionmax-500k"
sae_model_path = "conan1024hao/qwen2.5-vl-3b-sae-projection-mlp2-finevisionmax-500k"
model_with_sae = PeftSaeModel.from_pretrained(
    model,
    sae_model_path,
    adapter_name="default",
    low_cpu_mem_usage=True,
)

generated_ids = model_with_sae.generate(**inputs, max_new_tokens=128, do_sample=False)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print("============================================")
print("Model with SAE Output:")
print(output_text[0])
