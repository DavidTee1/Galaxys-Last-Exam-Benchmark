# Galaxy's Last Exam Benchmark

![Main Image](https://github.com/DavidTee1/Galaxys-Last-Exam-Benchmark/blob/main/Figures/Main_Image.png)

The **Galaxy's Last Exam Benchmark** is a challenging multimodal evaluation suite designed to push the boundaries of visual reasoning and metaphysical competence in AI models. It consists of a diverse set of tasks that require advanced perception, pattern recognition, and problem-solving skills beyond conventional datasets.

---

## Benchmark Description

Galaxy's Last Exam Benchmark tests AI models on complex multimodal reasoning tasks that integrate both visual and textual inputs. The tasks involve intricate spatial and logical deductions that demand an advanced understanding of visual representations and abstract reasoning.

### Benchmark Tasks

The benchmark includes several challenging tasks designed to evaluate a model's advanced reasoning capabilities:

1. **Line Intersection Counting:**  
   The model is provided with a chart featuring three distinct colors and is tasked with counting the number of times two lines intersect. 

2. **Letter Frequency Counting:**  
   The model must count the number of times a specific letter appears within an image.

3. **Cube Counting in 3D Structures:**  
   In this task, the model counts the number of individual cubes that comprise a larger three-dimensional structure.
   
4. **Wooden Slide Puzzle Moves Calculation:**  
   The model is challenged to determine how many moves it would take to reach a specific configuration in a wooden slide puzzle. This task assesses sequential planning and problem-solving skills. This task was inspired by [AlgoPuzzleVQA](https://github.com/declare-lab/LLM-PuzzleTest/tree/master/AlgoPuzzleVQA).

5. **Analog Clock Time Change Simulation:**  
   The model calculates the new positions of the clock hands on an analog clock after a given time change, demonstrating its ability to understand time-based transformations and spatial arrangements. This task was inspired by [AlgoPuzzleVQA](https://github.com/declare-lab/LLM-PuzzleTest/tree/master/AlgoPuzzleVQA).

---

## Training and Evaluation Data

- **Data Source:**  
  The benchmark consists entirely of **synthetically generated** data, ensuring a controlled evaluation environment. Scripts will be provided later to generate each dataset for reproducibility.

![Performance Comparison](https://github.com/DavidTee1/Galaxys-Last-Exam-Benchmark/blob/main/Figures/Performance_Comparison.png)

- **Prompt Format:**  
  The expected input format for prompts is:
  ```text
  <|image_1|> + user text
  ```

---

# Galactus: A Model for Galaxy's Last Exam Benchmark

Galactus is a state-of-the-art, multimodal language model fine-tuned from [microsoft/Phi-4-multimodal-instruct](https://huggingface.co/microsoft/Phi-4-multimodal-instruct) specifically to excel on the Galaxy's Last Exam Benchmark. It is optimized for advanced visual reasoning tasks that require metaphysical competence.

## Model Description

Galactus is designed to push the boundaries of visual and textual reasoning. It accepts image input along with text prompts to solve tasks that include, for example, calculating intersections of lines or simulating time changes on analog clocks. On the Galaxy's Last Exam Benchmark, this model outperforms comparable models from OpenAI and Gemini.

---

... (rest of the document remains unchanged)

## Intended Uses & Limitations

- **Intended Uses:**  
  - Handling complex visual reasoning tasks that require high metaphysical competence.  
  - Tasks that go beyond routine human problem-solving.

- **Limitations:**  
  - **Not** meant for standard everyday tasks or simple human interactions.  
  - Exclusively trained on the Galaxy's Last Exam Benchmark, so its generalization outside of this domain may be limited.

---

### Training Procedure

The fine-tuning process is centered on the vision components of the base Phi-4 model using LoRA adapters. Key aspects include:

- **Local Fine-Tuning:**  
  The script fine-tunes a locally downloaded Phi-4-Multimodal model using an image, prompt, and answer CSV file.

- **LoRA Adapters:**  
  Focus is on adapting vision components while keeping other parts of the model frozen.

- **Dropout Adjustments:**  
  Optional disabling of dropout layers for overfitting on smaller datasets.

- **Gradient Checkpointing:**  
  Enabled with specific configurations to optimize memory usage.

### Training Hyperparameters

Below are some of the hyperparameters used during training:

- **Epochs:** Approximately 252 epochs (as per checkpoint details)
- **Per Device Train Batch Size:** Specified via arguments
- **Gradient Checkpointing:** Enabled with `{use_reentrant: False}`
- **Gradient Accumulation Steps:** Configured based on dataset size and available GPUs
- **Optimizer:** `adamw_torch` with:
  - `adam_beta1`: 0.9
  - `adam_beta2`: 0.95
  - `adam_epsilon`: 1e-7
- **Learning Rate:** Specified via arguments
- **Weight Decay:** 0.0 (or as specified)
- **Save Strategy:** Steps-based saving every 10 steps
- **Evaluation Strategy:** Steps-based evaluation (if evaluation data is provided)
- **Max Gradient Norm:** 1.0
- **Learning Rate Scheduler:** Linear with 50 warmup steps

The model achieved **72% performance** on the Galaxy's Last Exam Benchmark.

### Framework Versions

- **Transformers:** 4.46.1
- **PyTorch:** 2.7.0.dev20250304+cu128
- **TorchVision:** 0.22.0.dev20250304+cu128
- **Tokenizers:** 0.20.3

---

## Repository File Structure

```
.
├── config.json                  # Configuration file containing API keys
├── prompts.csv                  # CSV file with [Prompt, Solution] for training/evaluation
├── results.csv                  # Output file from the sample evaluation script
├── evaluation_matrix.csv        # Detailed evaluation results across models
├── evaluation_summary.csv       # Performance summary across models
├── Images/                      # Directory containing images (named Image_1.jpg/png, etc.)
├── sample_eval.py               # Evaluation script for baseline models (OpenAI, Gemini, Claude)
├── finetune.py                  # Fine-tuning script for Phi-4-Multimodal on the Galaxy's Last Exam Benchmark
├── evaluate_model.py            # Evaluation script using GPT-4o-mini to assess model answers
└── evaluate_finetuned.py        # Evaluation script for the fine-tuned model on the Galactus benchmark
```

---

## Usage Instructions

### 1. Fine-Tuning

To fine-tune the model locally, run the `finetune.py` script. This script requires several arguments including the local paths for the model, processor, training CSV, evaluation CSV (optional), and the image directory. Example:

```bash
python finetune.py \
  --model_path /path/to/local/phi4-model \
  --processor_path /path/to/local/processor \
  --train_file prompts.csv \
  --eval_file eval_prompts.csv \
  --image_dir Images \
  --batch_size 4 \
  --num_train_epochs 30 \
  --learning_rate 1e-4
```

### 2. Evaluation Scripts

There are multiple scripts available for evaluating model performance:

- **`sample_eval.py`**: Evaluates the fine-tuned model on the Galactus benchmark by comparing outputs from different models (OpenAI, Gemini, Claude) on a set of prompts with associated images.
- **`evaluate_model.py`**: Uses the GPT-4o-mini model to evaluate if the predicted answer matches the expected solution. This script outputs detailed evaluation results (matrix and summary CSV files).
- **`evaluate_finetuned.py`**: Loads either the base or a fine-tuned model and evaluates it on a vision + text CSV dataset. Example:

```bash
python evaluate_finetuned.py \
  --finetuned_weights_path /path/to/fine-tuned-model \
  --data_file prompts.csv \
  --image_dir Images \
  --output_file evaluation_results
```

### 3. Configuration

Ensure that a valid `config.json` is in the repository root that provides API keys for:

- OpenAI
- Gemini
- Anthropic

---

## Dependencies

Install required packages using:

```bash
pip install scipy peft backoff transformers accelerate datasets torch pillow pandas tqdm openai google-genai anthropic
```

---

## Contributing

Contributions, issues, and feature requests are welcome. Check the issues page to contribute.

---

## License

This project is licensed under the **MIT License** – see the `LICENSE` file for details.

