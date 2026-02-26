import os
import pandas as pd
from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from PIL import Image
import copy
import torch
import warnings
from transformers import T5Tokenizer, T5ForConditionalGeneration
from tqdm import tqdm

warnings.filterwarnings("ignore")

pretrained = "lmms-labllama3-llava-next-8b"
model_name = "llava_llama3"
conv_template = "llava_llama_3"
device = "cuda:0"
device_map = {"": device}

image_folder = "../dataset/HAM10000/train"
metadata_file = "../dataset/HAM10000/train/raw.csv"
output_csv = "../dataset/HAM10000/train/metadata.csv"

tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name,
                                                                      device_map=device_map)
model.to(device).eval()
t5_model_name = "t5-large"
t5_tokenizer = T5Tokenizer.from_pretrained(t5_model_name)
t5_model = T5ForConditionalGeneration.from_pretrained(t5_model_name).to(device)

metadata = pd.read_csv(metadata_file)
results = []
counter = 0
for index, row in tqdm(metadata.iterrows(), total=len(metadata), desc="Generating descriptions"):
    filename = row['file_name']
    text = row['text']
    classname = text.split("of")[-1].strip()

    if filename.lower().endswith('.jpg'):
        image_path = os.path.join(image_folder, filename)
        if os.path.exists(image_path):
            try:
                image = Image.open(image_path)
                image_tensor = process_images([image], image_processor, model.config)
                image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]
                while True:
                    question = (
                        f"{DEFAULT_IMAGE_TOKEN}\n"
                        f"{text}. Please describe the dermoscopic image of {classname} using the following visual features: specific color, symmetry, border,shape, texture and dermoscopic patterns. Start with '{text}' and ensure the description does not exceed 100 words."
                    )

                    conv = copy.deepcopy(conv_templates[conv_template])
                    conv.append_message(conv.roles[0], question)
                    conv.append_message(conv.roles[1], None)
                    prompt_question = conv.get_prompt()

                    input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX,
                                                      return_tensors="pt").unsqueeze(0).to(device)
                    image_sizes = [image.size]

                    with torch.no_grad():
                        cont = model.generate(
                            input_ids,
                            images=image_tensor,
                            image_sizes=image_sizes,
                            do_sample=True,
                            temperature=0.8,
                            max_new_tokens=150,
                            top_k=50,
                            top_p=0.9,
                        )

                    text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)
                    generated_text = text_outputs[0]

                    t5_token_count = t5_tokenizer(generated_text, return_tensors="pt").input_ids.size(1)

                    if t5_token_count > 120:
                        print(f"Text too long ({t5_token_count} T5 tokens), simplifying...")
                        simplify_input = f"simplify: {generated_text}"
                        simplify_input_ids = t5_tokenizer.encode(simplify_input, return_tensors="pt", max_length=512,
                                                                 truncation=True).to(device)

                        with torch.no_grad():
                            simplify_outputs = t5_model.generate(
                                simplify_input_ids,
                                max_length=120,
                                min_length=90,
                                num_beams=4,
                                early_stopping=True
                            )
                        simplified_text = t5_tokenizer.decode(simplify_outputs[0], skip_special_tokens=True)
                    else:
                        simplified_text = generated_text

                    if simplified_text.startswith(f"\nA dermoscopic image of"):
                        final_t5_token_count = t5_tokenizer(simplified_text, return_tensors="pt").input_ids.size(1)
                        print(f"Final Text Accepted ({final_t5_token_count} T5 tokens): {simplified_text}")
                        break
                    else:
                        print(f"Text did not meet requirements, regenerating: {simplified_text}")

                results.append({"file_name": filename, "text": simplified_text})
                counter += 1
                print(f"Processed {counter}/{len(metadata)}")
            except Exception as e:
                print(f"\nAn error occurred while processing {filename}: {e}")
        else:
            print(f"\nImage not found: {image_path}")

if results:
    df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False, encoding='utf-8')
    print(f"\nProcessing complete. Final results saved to {output_csv}")
