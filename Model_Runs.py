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
import base64
import logging
import pandas as pd
import requests
from PIL import Image

import openai
from google import genai
import anthropic

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

with open("config.json", "r") as f:
    config = json.load(f)

OPENAI_API_KEY = config.get("openai")
GEMINI_API_KEY = config.get("gemini")
ANTHROPIC_API_KEY = config.get("anthropic")

openai.api_key = OPENAI_API_KEY
gemini_client = genai.Client(api_key=GEMINI_API_KEY)
claude_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


def encode_image(image_path):
    """Encode an image in Base64."""
    try:
        with open(image_path, "rb") as image_file:
            data = base64.b64encode(image_file.read()).decode("utf-8")
        logging.debug(f"Image encoded successfully: {image_path}")
        return data
    except Exception as e:
        logging.error(f"Error encoding image {image_path}: {e}")
        raise

def test_openai_model(prompt, image_path, model):
    """Test an OpenAI model by sending a text prompt with an image."""
    try:
        logging.debug(f"Testing OpenAI model {model} for prompt: {prompt}")
        b64_image = encode_image(image_path)
        response = openai.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}}
                    ]
                }
            ],
        )
        output = response.choices[0].message
        logging.debug(f"OpenAI model {model} output: {output}")
        return output
    except Exception as e:
        error_msg = f"Error in OpenAI model ({model}): {e}"
        logging.error(error_msg)
        return error_msg

def test_gemini_model(prompt, image_path, model):
    """Test a Gemini model by sending a PIL image along with the prompt."""
    try:
        logging.debug(f"Testing Gemini model {model} for prompt: {prompt}")
        pil_image = Image.open(image_path)
        response = gemini_client.models.generate_content(
            model=model,
            contents=[prompt, pil_image]
        )
        output = response.text
        logging.debug(f"Gemini model {model} output: {output}")
        return output
    except Exception as e:
        error_msg = f"Error in Gemini model ({model}): {e}"
        logging.error(error_msg)
        return error_msg

def test_claude_model(prompt, image_path, model):
    """Test the Claude model by sending a Base64 encoded image along with the prompt."""
    try:
        logging.debug(f"Testing Claude model {model} for prompt: {prompt}")
        b64_data = encode_image(image_path)
        message = claude_client.messages.create(
            model=model,
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": b64_data,
                            },
                        },
                        {"type": "text", "text": prompt}
                    ],
                }
            ],
        )
        output = message.text if hasattr(message, "text") else str(message)
        logging.debug(f"Claude model {model} output: {output}")
        return output
    except Exception as e:
        error_msg = f"Error in Claude model ({model}): {e}"
        logging.error(error_msg)
        return error_msg


def main():
    image_dir = "Images"
    if not os.path.isdir(image_dir):
        error_msg = f"Image directory '{image_dir}' not found."
        logging.error(error_msg)
        raise FileNotFoundError(error_msg)
    else:
        logging.debug(f"Image directory '{image_dir}' verified.")

    input_csv = "prompts.csv"  
    try:
        df = pd.read_csv(input_csv)
        logging.debug(f"Loaded prompts from {input_csv}")
    except Exception as e:
        error_msg = f"Error reading CSV file {input_csv}: {e}"
        logging.error(error_msg)
        raise

    results = []
    openai_models = ["o1", "gpt-4o", "gpt-4o-mini"]
    gemini_models = ["gemini-1.5-flash", "gemini-1.5-pro"]
    claude_model = "claude-3-7-sonnet-20250219"

    for index, row in df.iterrows():
        prompt = row[0]
        image_number = index + 1
        image_filename = f"Image_{image_number}.png"  
        image_path = os.path.join(image_dir, image_filename)
        
        row_data = {"Prompt": prompt}

        if not os.path.exists(image_path):
            error_text = f"Image file {image_filename} not found."
            logging.error(error_text)
            for model in openai_models + gemini_models + [claude_model]:
                row_data[model] = error_text
        else:
            for model in openai_models:
                result = test_openai_model(prompt, image_path, model)
                row_data[model] = result
            for model in gemini_models:
                result = test_gemini_model(prompt, image_path, model)
                row_data[model] = result
            result = test_claude_model(prompt, image_path, claude_model)
            row_data[claude_model] = result

        results.append(row_data)

    result_df = pd.DataFrame(results)
    output_csv = "results.csv"
    try:
        result_df.to_csv(output_csv, index=False)
        logging.debug(f"Test results saved to {output_csv}")
    except Exception as e:
        logging.error(f"Error writing CSV file {output_csv}: {e}")

if __name__ == "__main__":
    main()
