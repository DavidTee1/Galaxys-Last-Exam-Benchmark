#!/usr/bin/env python3
"""
Phi-4-Multimodal Vision Model Local Fine-Tuning

This script fine-tunes a locally downloaded Phi-4-Multimodal model on the Galaxy's Last Exam with image, prompt, and answer format.

Requirements:
- scipy==1.15.1
- peft==0.13.2
- backoff==2.2.1
- transformers==4.46.1
- accelerate==1.3.0
- datasets
- torch
- pillow
"""

import argparse
import json
import os
import logging
from pathlib import Path

import torch
import pandas as pd
from accelerate import Accelerator
from accelerate.utils import gather_object
from datasets import Dataset
from PIL import Image
from torch.utils.data import Dataset as TorchDataset, ConcatDataset
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    BatchFeature,
    Trainer,
    TrainingArguments,
)
from transformers import EarlyStoppingCallback


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("phi4_finetune.log")
    ]
)
logger = logging.getLogger(__name__)

_IGNORE_INDEX = -100
_MAX_TRAINING_LENGTH = 8192


class CustomVisionDataset(TorchDataset):
    """
    Custom dataset for vision fine-tuning with simple image, prompt, answer format
    """
    def __init__(self, processor, data_file, image_dir, split="train"):
        """
        Args:
            processor: The Phi-4 processor
            data_file: Path to CSV file with dataset annotations
            image_dir: Directory containing the images
            split: "train" or "test"
        """
        self.processor = processor
        self.image_dir = Path(image_dir)
        
        logger.info(f"Loading {split} dataset from {data_file}")
        df = pd.read_csv(data_file)
        
        if 'image_filename' not in df.columns:
            logger.info("Creating image_filename column based on 'Image_N' naming convention")
            df['image_filename'] = [f"Image_{i+1}.jpg" for i in range(len(df))]
            
        self.annotations = Dataset.from_pandas(df)
        logger.info(f"Loaded {len(self.annotations)} {split} examples")
        
        self.split = split

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        """Process a dataset example"""
        annotation = self.annotations[idx]
    
        if 'image_filename' not in annotation:
            img_idx = idx % 50 + 1  
            image_filename = f"Image_{img_idx}.jpg"
        else:
            image_filename = annotation['image_filename']
        
            img_idx = int(image_filename.split('_')[1].split('.')[0]) if '_' in image_filename else idx % 50 + 1

        image_path = self.image_dir / image_filename

        if not image_path.exists():
            for ext in ['.png', '.jpg', '.jpeg', '.JPG', '.JPEG', '.PNG']:
                alt_path = self.image_dir / f"Image_{img_idx}{ext}"
                if alt_path.exists():
                    image_path = alt_path
                    break
    
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path} (tried Image_{img_idx}.jpg etc.)")
        
        image = Image.open(image_path).convert('RGB')
        image = image.resize((448, 448), resample=Image.BICUBIC)


        prompt = annotation['Prompt']
        if "<|image_1|>" not in prompt:
            prompt = "<|image_1|>\n" + prompt

        answer = annotation['Solution']    
        inputs = self.processor(
            text=prompt,
            images=[image],
            return_tensors="pt"
        )

        if idx == 0:
            logger.info(f"Debug - Input shape: {inputs.input_ids.shape}")
            logger.info(f"Debug - Image embeddings shape: {inputs.input_image_embeds.shape}")
    
        if self.split == "train":
            formatted_answer = f"{answer}<|end|><|endoftext|>"
            answer_ids = self.processor.tokenizer(formatted_answer, return_tensors='pt').input_ids

            input_ids = torch.cat([inputs.input_ids, answer_ids], dim=1)
            labels = torch.full_like(input_ids, _IGNORE_INDEX)
            labels[:, -answer_ids.shape[1]:] = answer_ids

            """
            # Truncate if too long
            if input_ids.size(1) > _MAX_TRAINING_LENGTH:
                input_ids = input_ids[:, :_MAX_TRAINING_LENGTH]
                labels = labels[:, :_MAX_TRAINING_LENGTH]
                if torch.all(labels == _IGNORE_INDEX).item():
                    # Make sure loss compute won't fail
                    labels[:, -1] = self.processor.tokenizer.eos_token_id
            """
            return {
                'input_ids': input_ids,
                'labels': labels,
                'input_image_embeds': inputs.input_image_embeds,
                'image_attention_mask': inputs.image_attention_mask,
                'image_sizes': inputs.image_sizes,
            }
        else:
            return {
                'id': str(annotation.get('id', idx)),
                'input_ids': inputs.input_ids,
                'input_image_embeds': inputs.input_image_embeds,
                'image_attention_mask': inputs.image_attention_mask,
                'image_sizes': inputs.image_sizes,
                'answer': answer,
            }

def pad_sequence(sequences, padding_side='right', padding_value=0):
    """
    Pad a list of sequences to the same length.
    sequences: list of tensors in [seq_len, *] shape
    """
    assert padding_side in ['right', 'left']
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    max_len = max(len(seq) for seq in sequences)
    batch_size = len(sequences)
    output = sequences[0].new_full((batch_size, max_len) + trailing_dims, padding_value)
    for i, seq in enumerate(sequences):
        length = seq.size(0)
        if padding_side == 'right':
            output.data[i, :length] = seq
        else:
            output.data[i, -length:] = seq
    return output


def cat_with_pad(tensors, dim, padding_value=0):
    """
    cat along dim, while pad to max for all other dims
    """
    ndim = tensors[0].dim()
    assert all(
        t.dim() == ndim for t in tensors[1:]
    ), 'All tensors must have the same number of dimensions'

    out_size = [max(t.shape[i] for t in tensors) for i in range(ndim)]
    out_size[dim] = sum(t.shape[dim] for t in tensors)
    output = tensors[0].new_full(out_size, padding_value)

    index = 0
    for t in tensors:
        slices = [slice(0, t.shape[d]) for d in range(ndim)]
        slices[dim] = slice(index, index + t.shape[dim])

        output[slices] = t
        index += t.shape[dim]

    return output


def vision_collate_fn(batch):
    """Collate function for training data"""
    input_ids_list = []
    labels_list = []
    input_image_embeds_list = []
    image_attention_mask_list = []
    image_sizes_list = []
    
    for inputs in batch:
        input_ids_list.append(inputs['input_ids'][0])
        labels_list.append(inputs['labels'][0])
        input_image_embeds_list.append(inputs['input_image_embeds'])
        image_attention_mask_list.append(inputs['image_attention_mask'])
        image_sizes_list.append(inputs['image_sizes'])

    input_ids = pad_sequence(input_ids_list, padding_side='right', padding_value=0)
    labels = pad_sequence(labels_list, padding_side='right', padding_value=0)
    attention_mask = (input_ids != 0).long()
    input_image_embeds = cat_with_pad(input_image_embeds_list, dim=0)
    image_attention_mask = cat_with_pad(image_attention_mask_list, dim=0)
    image_sizes = torch.cat(image_sizes_list)

    return BatchFeature(
        {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask,
            'input_image_embeds': input_image_embeds,
            'image_attention_mask': image_attention_mask,
            'image_sizes': image_sizes,
            'input_mode': 1,  
        }
    )


def vision_eval_collate_fn(batch):
    """Collate function for evaluation data"""
    input_ids_list = []
    input_image_embeds_list = []
    image_attention_mask_list = []
    image_sizes_list = []
    all_ids = []
    all_answers = []
    
    for inputs in batch:
        input_ids_list.append(inputs['input_ids'][0])
        input_image_embeds_list.append(inputs['input_image_embeds'])
        image_attention_mask_list.append(inputs['image_attention_mask'])
        image_sizes_list.append(inputs['image_sizes'])
        all_ids.append(inputs['id'])
        all_answers.append(inputs['answer'])

    input_ids = pad_sequence(input_ids_list, padding_side='left', padding_value=0)
    attention_mask = (input_ids != 0).long()
    input_image_embeds = cat_with_pad(input_image_embeds_list, dim=0)
    image_attention_mask = cat_with_pad(image_attention_mask_list, dim=0)
    image_sizes = torch.cat(image_sizes_list)

    return (
        all_ids,
        all_answers,
        BatchFeature(
            {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'input_image_embeds': input_image_embeds,
                'image_attention_mask': image_attention_mask,
                'image_sizes': image_sizes,
                'input_mode': 1,  
            }
        ),
    )

#had some issues with flash attention on cu128
def load_model_from_local(model_path, use_flash_attention=False):
    """
    Load Phi-4-Multimodal model from local path
    
    Args:
        model_path: Path to the local model directory
        use_flash_attention: Whether to use flash attention
        
    Returns:
        Loaded model
    """
    logger.info(f"Loading model from local path: {model_path}")
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16 if use_flash_attention else torch.float32,
            _attn_implementation='flash_attention_2' if use_flash_attention else 'sdpa',
            trust_remote_code=True,
            local_files_only=True,  
        ).to('cuda')
        
        del model.model.embed_tokens_extend.audio_embed 
        for layer in model.model.layers:
            del layer.mlp.down_proj.lora_A.speech
            del layer.mlp.down_proj.lora_B.speech
            del layer.mlp.gate_up_proj.lora_A.speech
            del layer.mlp.gate_up_proj.lora_B.speech
            del layer.self_attn.o_proj.lora_A.speech
            del layer.self_attn.o_proj.lora_B.speech
            del layer.self_attn.qkv_proj.lora_A.speech
            del layer.self_attn.qkv_proj.lora_B.speech
            
        logger.info("Model loaded successfully")
        return model
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


def load_processor_from_local(processor_path, dynamic_hd=36):
    """
    Load processor from local path
    
    Args:
        processor_path: Path to the local processor directory
        dynamic_hd: Number of maximum image crops
        
    Returns:
        Loaded processor
    """
    logger.info(f"Loading processor from local path: {processor_path}")
    
    try:
        processor = AutoProcessor.from_pretrained(
            processor_path,
            trust_remote_code=True,
            dynamic_hd=dynamic_hd,
            local_files_only=True, 
        )
        logger.info("Processor loaded successfully")
        return processor
        
    except Exception as e:
        logger.error(f"Failed to load processor: {e}")
        raise


@torch.no_grad()
def evaluate(
    model, processor, eval_dataset, save_path=None, disable_tqdm=False, eval_batch_size=1
):
    """
    Evaluate the model on the evaluation dataset
    
    Args:
        model: The fine-tuned model
        processor: The text processor
        eval_dataset: Evaluation dataset
        save_path: Path to save evaluation results
        disable_tqdm: Whether to disable progress bar
        eval_batch_size: Batch size for evaluation
        
    Returns:
        Accuracy of the model
    """
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))

    model.eval()
    all_answers = []
    all_generated_texts = []

    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=eval_batch_size,
        collate_fn=vision_eval_collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=8,
        prefetch_factor=2,
        pin_memory=True,
    )
    
    for ids, answers, inputs in tqdm(
        eval_dataloader, disable=(rank != 0) or disable_tqdm, desc='Running evaluation'
    ):
        all_answers.extend({'id': i, 'answer': a.strip().lower()} for i, a in zip(ids, answers))

        inputs = inputs.to(f'cuda:{local_rank}')
        generated_ids = model.generate(
            **inputs, eos_token_id=processor.tokenizer.eos_token_id, max_new_tokens=64
        )

        input_len = inputs.input_ids.size(1)
        generated_texts = processor.batch_decode(
            generated_ids[:, input_len:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        all_generated_texts.extend(
            {'id': i, 'generated_text': g.strip().lower()} for i, g in zip(ids, generated_texts)
        )

    all_answers = gather_object(all_answers)
    all_generated_texts = gather_object(all_generated_texts)

    if rank == 0:
        assert len(all_answers) == len(all_generated_texts)

        exact_matches = 0
        for a, g in zip(all_answers, all_generated_texts):
            answer = a['answer'].strip().lower()
            generated = g['generated_text'].strip().lower()
            if answer == generated:
                exact_matches += 1
                
        acc = exact_matches / len(all_answers)

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'w') as f:
                save_dict = {
                    'answers': all_answers,
                    'generated_texts': all_generated_texts,
                    'accuracy': acc,
                }
                json.dump(save_dict, f)

        return acc
    return None


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Phi-4-Multimodal on custom dataset")
    
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='Local path to the Phi-4 model directory'
    )
    parser.add_argument(
        '--processor_path',
        type=str,
        help='Local path to the processor directory (defaults to model_path)'
    )
    
    parser.add_argument(
        '--train_file',
        type=str,
        required=True,
        help='CSV file containing training data (image_filename, prompt, answer)'
    )
    parser.add_argument(
        '--eval_file',
        type=str,
        help='CSV file containing evaluation data (same format as train_file)'
    )
    parser.add_argument(
        '--image_dir',
        type=str,
        required=True,
        help='Directory containing all the images referenced in CSV files'
    )
    
    parser.add_argument('--use_flash_attention', action='store_true', help='Use Flash Attention')
    parser.add_argument('--output_dir', type=str, default='./output/', help='Output directory')
    parser.add_argument('--batch_size', type=int, default=4, help='Global batch size')
    parser.add_argument(
        '--gradient_accumulation_steps',
        type=int,
        default=None,  
        help='Number of gradient accumulation steps'
    )
    parser.add_argument(
        '--batch_size_per_gpu',
        type=int,
        default=1,
        help='Batch size per GPU (adjust to fit in GPU memory)'
    )
    parser.add_argument(
        '--dynamic_hd',
        type=int,
        default=36,
        help='Number of maximum image crops'
    )
    parser.add_argument(
        '--num_train_epochs', 
        type=int, 
        default=30, 
        help='Number of training epochs'
    )
    parser.add_argument(
        '--learning_rate', 
        type=float, 
        default=1e-4, 
        help='Learning rate'
    )
    parser.add_argument(
        '--weight_decay', 
        type=float, 
        default=0.0, 
        help='Weight decay'
    )
    parser.add_argument(
        '--no_tqdm', 
        dest='tqdm', 
        action='store_false', 
        help='Disable tqdm progress bars'
    )
    parser.add_argument(
        '--skip_eval', 
        action='store_true', 
        help='Skip evaluation steps'
    )
    parser.add_argument(
        '--early_stopping_patience', 
        type=int, 
        default=100, 
        help='Early stopping patience'
    )
    parser.add_argument(
        '--repeat_dataset', 
        type=int, 
        default=1, 
        help='Repeat small dataset this many times'
    )
    parser.add_argument(
        '--disable_dropout', 
        action='store_true',
        help='Disable all dropout layers for better overfitting'
    )
    
    args = parser.parse_args()
    
    if not args.processor_path:
        args.processor_path = args.model_path
    
    accelerator = Accelerator()
    
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    
    if rank == 0:
        logger.info(f"Training on {world_size} GPUs")
        logger.info(f"Model path: {args.model_path}")
        logger.info(f"Processor path: {args.processor_path}")
        logger.info(f"Training data: {args.train_file}")
        logger.info(f"Evaluation data: {args.eval_file if args.eval_file else 'None'}")
        logger.info(f"Image directory: {args.image_dir}")
        logger.info(f"Output directory: {args.output_dir}")
    
    if not os.path.exists(args.image_dir):
        raise ValueError(f"Image directory does not exist: {args.image_dir}")
    
    if not os.path.exists(args.train_file):
        raise ValueError(f"Training file does not exist: {args.train_file}")
    
    if args.eval_file and not os.path.exists(args.eval_file):
        raise ValueError(f"Evaluation file does not exist: {args.eval_file}")
    
    with accelerator.local_main_process_first():
        processor = load_processor_from_local(
            args.processor_path,
            dynamic_hd=args.dynamic_hd,
        )
        model = load_model_from_local(
            args.model_path,
            use_flash_attention=args.use_flash_attention,
        )
    
    model.set_lora_adapter('vision')
    for param in model.model.embed_tokens_extend.image_embed.parameters():
        param.requires_grad = True
    
    if args.disable_dropout:
        logger.info("Disabling dropout layers for better overfitting")
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.0
    
        for layer in model.model.layers:
            if hasattr(layer.self_attn, 'dropout'):
                layer.self_attn.dropout.p = 0.0
            if hasattr(layer.mlp, 'dropout'):
                layer.mlp.dropout.p = 0.0
            if hasattr(layer.self_attn, 'attn_dropout'):
                layer.self_attn.attn_dropout.p = 0.0
    
    train_dataset = CustomVisionDataset(
        processor=processor, 
        data_file=args.train_file,
        image_dir=args.image_dir,
        split="train"
    )
    
    if args.repeat_dataset > 1:
        logger.info(f"Repeating training dataset {args.repeat_dataset} times")
        original_length = len(train_dataset)
        train_datasets = [train_dataset]
        for _ in range(args.repeat_dataset - 1):
            train_datasets.append(CustomVisionDataset(
                processor=processor, 
                data_file=args.train_file,
                image_dir=args.image_dir,
                split="train"
            ))    
        train_dataset = ConcatDataset(train_datasets)
    
        logger.info(f"Expanded dataset from {original_length} to {len(train_dataset)} examples")
    
    eval_dataset = None
    if not args.skip_eval and args.eval_file:
        eval_dataset = CustomVisionDataset(
            processor=processor,
            data_file=args.eval_file,
            image_dir=args.image_dir,
            split="test"
        )
    
    num_gpus = accelerator.num_processes
    
    if args.gradient_accumulation_steps is None:
        if len(train_dataset) <= 50:
            grad_acc_multiplier = max(1, 50 // len(train_dataset))
            args.gradient_accumulation_steps = max(2, grad_acc_multiplier * (args.batch_size // (num_gpus * args.batch_size_per_gpu)))
        else:
            args.gradient_accumulation_steps = args.batch_size // (num_gpus * args.batch_size_per_gpu)
        
    logger.info(f"Using gradient accumulation steps: {args.gradient_accumulation_steps}")
    
    assert (
        args.batch_size % (num_gpus * args.batch_size_per_gpu) == 0
    ), 'Batch size must be divisible by (number of GPUs * batch size per GPU)'
    
    if args.use_flash_attention:
        fp16 = False
        bf16 = True
    else:
        fp16 = True
        bf16 = False
    
    # gradient checkpointing won't work here
    training_args = TrainingArguments(
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size_per_gpu,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        optim='adamw_torch',
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_epsilon=1e-7,
        learning_rate=args.learning_rate,
        weight_decay=0.0 if args.weight_decay == 0.0 else args.weight_decay,
        save_strategy='steps',
        save_steps=10,  
        eval_steps=10 if eval_dataset else None, 
        evaluation_strategy='steps' if eval_dataset else 'no',
        load_best_model_at_end=True if eval_dataset else False,
        max_grad_norm=1.0,
        lr_scheduler_type='linear',
        warmup_steps=50,
        logging_steps=10,
        output_dir=args.output_dir,
        save_total_limit=2,
        save_only_model=True,
        bf16=bf16,
        fp16=fp16,
        remove_unused_columns=False,
        report_to='none',
        deepspeed=None,
        disable_tqdm=not args.tqdm,
        dataloader_num_workers=4,
        ddp_find_unused_parameters=True, 
    )
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if not args.skip_eval and eval_dataset:
        logger.info("Evaluating model before fine-tuning...")
        acc = evaluate(
            model,
            processor,
            eval_dataset,
            save_path=os.path.join(args.output_dir, 'eval_before.json'),
            disable_tqdm=not args.tqdm,
            eval_batch_size=args.batch_size_per_gpu,
        )
        if accelerator.is_main_process and acc is not None:
            logger.info(f'Accuracy before fine-tuning: {acc:.4f}')
    
    callbacks = []
    if eval_dataset and args.early_stopping_patience > 0:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=args.early_stopping_patience,
                early_stopping_threshold=1e-6
            )
        )
    
    logger.info("Starting model fine-tuning...")
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=vision_collate_fn,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=callbacks
    )
    
    trainer.train()
    trainer.save_model()
    accelerator.wait_for_everyone()
    
    if not args.skip_eval and eval_dataset:
        logger.info("Evaluating model after fine-tuning...")
        

        del model
        del trainer
        __import__('gc').collect()
        torch.cuda.empty_cache()
        
        model = load_model_from_local(
            args.output_dir,
            use_flash_attention=args.use_flash_attention,
        )
        
        acc = evaluate(
            model,
            processor,
            eval_dataset,
            save_path=os.path.join(args.output_dir, 'eval_after.json'),
            disable_tqdm=not args.tqdm,
            eval_batch_size=args.batch_size_per_gpu,
        )
        if accelerator.is_main_process and acc is not None:
            logger.info(f'Accuracy after fine-tuning: {acc:.4f}')
    
    logger.info("Fine-tuning process completed successfully!")


if __name__ == '__main__':
    main()