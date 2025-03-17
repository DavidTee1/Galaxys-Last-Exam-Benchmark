"""
sample eval script for finetuned model on Galactus benchmark
- Loads base model or phi4 from huggingface
- Loads our fine-tuned checkpoint for updated weights
- Uses a GenerationConfig as a workaround
- Evaluates on a CSV with [Prompt, Solution].
- Looks for images named Image_{index}.jpg/png etc. Refactor if you don't set it up in this manner
"""

import os
import json
import torch
import argparse
import pandas as pd
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    AutoModelForCausalLM,
    GenerationConfig
)


def evaluate_finetuned_model(
    finetuned_weights_path,
    data_file,
    image_dir,
    output_file=None,
    original_model_path=None
):
    """
    Evaluate a fine-tuned model on a vision + text CSV dataset.
    
    Args:
        finetuned_weights_path: Folder with config.json, model.safetensors,
        data_file: Path to CSV with [Prompt, Solution]
        image_dir: Directory containing images for each example.
        output_file: Path to save final JSON + partial CSV results.
        original_model_path: Path to original base model
    """
    # ------------------------------------------------
    # 1) load base or HF model & processor
    # ------------------------------------------------
    hf_model_id = "Phi-4-multimodal-instruct"
    
    if original_model_path and os.path.exists(original_model_path):
        print(f"Loading base model from local path: {original_model_path}")
        processor = AutoProcessor.from_pretrained(
            original_model_path,
            trust_remote_code=True,
            local_files_only=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            original_model_path,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
            _attn_implementation="sdpa",
            local_files_only=True
        )
    else:
        print(f"Loading base model from Hugging Face: {hf_model_id}")
        processor = AutoProcessor.from_pretrained(
            hf_model_id,
            trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            hf_model_id,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
            _attn_implementation="sdpa"
        )
    
    print("Successfully loaded base model and processor.")
    
    # ------------------------------------------------
    # 2) attempt to load the fine-tuned checkpoint
    # ------------------------------------------------
    try:
        print(f"Loading fine-tuned model from: {finetuned_weights_path}")
        model = AutoModelForCausalLM.from_pretrained(
            finetuned_weights_path,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
            _attn_implementation="sdpa"
        )
        
        try:
            new_processor = AutoProcessor.from_pretrained(
                finetuned_weights_path,
                trust_remote_code=True
            )
            processor = new_processor
            print("Successfully loaded fine-tuned processor.")
        except Exception as proc_error:
            print(f"Warning: Could not load processor from {finetuned_weights_path}: {proc_error}")
            print("Using original processor instead.")
        
        print("Successfully loaded fine-tuned model.")
    except Exception as e:
        print(f"Failed to load fine-tuned model from {finetuned_weights_path}: {e}")
        print("Falling back to base model weights and processor.")
    
    # ------------------------------------------------
    # 2.5) gen config load
    # ------------------------------------------------
    try:
        gen_config = GenerationConfig.from_pretrained(finetuned_weights_path, "generation_config.json")
        print("Loaded generation_config.json from fine-tuned checkpoint.")
    except:
        print("No generation_config.json found. Using default GenerationConfig.")
        gen_config = GenerationConfig()
    
    # ------------------------------------------------
    # 3) eval setup
    # ------------------------------------------------
    model.eval()
    print(f"Loading dataset from {data_file}")
    df = pd.read_csv(data_file)
    
    if 'Prompt' not in df.columns or 'Solution' not in df.columns:
        raise ValueError("CSV must have columns: Prompt, Solution.")
    
    results = []
    correct_count = 0
    total_count = len(df)
    
    # ------------------------------------------------
    # 4) Process each example
    # ------------------------------------------------
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
        prompt = row['Prompt']
        expected = row['Solution'].strip()
        
        image_filename = f"Image_{i+1}"
        found_image = False
        image_path = None
        
        for ext in ['.jpg', '.png', '.jpeg', '.JPG', '.PNG']:
            candidate = os.path.join(image_dir, image_filename + ext)
            if os.path.exists(candidate):
                image_path = candidate
                found_image = True
                break
        
        if not found_image:
            for filename in os.listdir(image_dir):
                if (filename.startswith(f"Image_{i+1}") or
                    filename == f"{i+1}.jpg" or
                    filename == f"{i+1}.png"):
                    image_path = os.path.join(image_dir, filename)
                    found_image = True
                    break
        
        if not found_image or not image_path:
            print(f"Error: Could not find image for example {i+1}")
            results.append({
                'image_filename': f"{image_filename}.*",
                'prompt': prompt,
                'expected': expected,
                'prediction': "ERROR: Image not found",
                'is_correct': False
            })
            continue
        
        try:
            # ------------------------------------------------
            # 4.1) Prepare the prompt & inputs
            # ------------------------------------------------
            # Microsoft approach for single-turn:
            # <|image_1|> + user text
            user_prompt = f"<|image_1|>{prompt}"
            
            image = Image.open(image_path).convert("RGB")
            
            inputs = processor(
                text=user_prompt,
                images=image,
                return_tensors="pt"
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items() if v is not None}
            
            # ------------------------------------------------
            # 4.2) Generate using GenerationConfig
            # ------------------------------------------------\
            # is_correct added for user simplicity - I would take another pass to validate though

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    generation_config=gen_config,
                    max_new_tokens=128 
                )
            
            new_tokens = output_ids[:, inputs['input_ids'].shape[1]:]
            
            output_text = processor.tokenizer.decode(
                new_tokens[0],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            
            prediction = output_text.strip()
            
            is_correct = (prediction.lower() == expected.lower())
            if is_correct:
                correct_count += 1
            
            result = {
                'image_filename': os.path.basename(image_path),
                'prompt': prompt,
                'expected': expected,
                'prediction': prediction,
                'is_correct': is_correct
            }
            results.append(result)
            
            if output_file:
                csv_output_file = output_file if output_file.lower().endswith('.csv') else output_file + ".csv"
            else:
                csv_output_file = "evaluation_results_FT.csv"
            pd.DataFrame(results).to_csv(csv_output_file, index=False)
            
            print(f"\nExample {i+1}:")
            print(f"Image: {os.path.basename(image_path)}")
            print(f"Prompt: {prompt}")
            print(f"Expected: {expected}")
            print(f"Prediction: {prediction}")
            print(f"Correct: {'Yay!' if is_correct else 'Oops!'}")
        
        except Exception as e:
            print(f"\nError processing example {i+1}: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'image_filename': os.path.basename(image_path),
                'prompt': prompt,
                'expected': expected,
                'prediction': f"ERROR: {str(e)}",
                'is_correct': False
            })
    
def main():
    parser = argparse.ArgumentParser(description="Evaluate Phi-4-Multimodal fine-tuned model")
    parser.add_argument("--original_model_path",
                        help="Path to original/base model (if not provided, loads from HF)")
    parser.add_argument("--finetuned_weights_path", required=True,
                        help="Path to fine-tuned model folder (with config.json, model.safetensors, etc.)")
    parser.add_argument("--data_file", required=True,
                        help="Path to CSV data file with Prompt and Solution columns")
    parser.add_argument("--image_dir", required=True,
                        help="Directory containing the images")
    parser.add_argument("--output_file",
                        help="Path to save evaluation results (JSON + partial CSV dumps)")
    
    args = parser.parse_args()
    
    evaluate_finetuned_model(
        args.finetuned_weights_path,
        args.data_file,
        args.image_dir,
        args.output_file,
        args.original_model_path
    )

if __name__ == "__main__":
    main()

