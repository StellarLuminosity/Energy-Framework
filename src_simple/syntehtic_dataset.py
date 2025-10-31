import torch
import datasets
from tqdm import tqdm
from transformers import AutoModelForCausalLM
import config
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import os
from simple_config import config
from simple_utils import fix_seed

fix_seed(config.seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = datasets.load_from_disk(config.dataset_path)
dataloader = DataLoader(dataset["train"].select(range(100)), batch_size=2)
ds = {"input_ids": [], "attention_mask": [], "labels": []}

teacher_model = AutoModelForCausalLM.from_pretrained(
    config.teacher_model_name,
    torch_dtype=torch.bfloat16,
)
teacher_model = teacher_model.to(device)
teacher_model.eval()

print("\n=== GENERATING TEACHER LOGITS ===")
for batch in tqdm(dataloader):
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)

    # Append the original inputs directly
    ds["input_ids"].extend(input_ids.cpu().unbind())
    ds["attention_mask"].extend(attention_mask.cpu().unbind())

    with torch.no_grad():
        generation_output = teacher.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            temperature=0.5,
            max_new_tokens=1024,
            return_dict_in_generate=True,
            output_scores=True,
        )

        generated_sequences = generation_output.sequences
        gen_only = generated_sequences[:, input_ids.shape[1] :]

        # pad = torch.full((gen_only.size(0), input_ids.size(1)), fill_value=-100, dtype=gen_only.dtype, device=gen_only.device)
        # labels = torch.cat([pad, gen_only], dim=1)

        # Create labels tensor with correct padding
        # The labels should be the same length as the input_ids + gen_only
        # We pad the original input_ids part with -100
        max_len = generated_sequences.shape[1]
        labels = torch.full((input_ids.size(0), max_len), fill_value=-100, dtype=torch.long, device=gen_only.device)
        
        # Fill the generated part with the actual tokens
        labels[:, input_ids.size(1):] = gen_only

        ds["labels"].extend(labels.cpu().unbind())

        del generation_output, generated_sequences, gen_only, labels, input_ids, attention_mask
        torch.cuda.empty_cache()



def generate_synthetic_dataset():
    # ----------------------------------
    # Setup and Initialization
    # ----------------------------------
    fix_seed(config.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    start_time = time.time()
    main_print(f"Starting synthetic dataset generation at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    main_print(f"Using device: {device}")
    
    # ----------------------------------
    # Load Dataset
    # ----------------------------------
    main_print("Loading source dataset...")
    try:
        dataset = datasets.load_from_disk(config.dataset_path)
        main_print(f"Loaded dataset with {len(dataset['train'])} training samples")
    except Exception as e:
        main_print(f"Error loading dataset: {e}")
        raise
    
    # Use a subset for synthetic generation (configurable)
    num_samples = getattr(config, 'synthetic_samples', 1000)
    subset_dataset = dataset["train"].select(range(min(num_samples, len(dataset['train']))))
    main_print(f"Using {len(subset_dataset)} samples for synthetic generation")
    
    # ----------------------------------
    # Create DataLoader
    # ----------------------------------
    dataloader = DataLoader(subset_dataset, batch_size=getattr(config, 'synthetic_batch_size', 2), shuffle=False)
    ds = {"input_ids": [], "attention_mask": [], "labels": []}
    
    # ----------------------------------
    # Load Teacher Model
    # ----------------------------------
    main_print(f"Loading teacher model: {config.teacher_model_name}")
    try:
        teacher_model = AutoModelForCausalLM.from_pretrained(
            config.teacher_model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        teacher_model.eval()
        main_print(f"Teacher model loaded successfully")
    except Exception as e:
        main_print(f"Error loading teacher model: {e}")
        raise
    
    # ----------------------------------
    # Load Tokenizer
    # ----------------------------------
    try:
        tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        main_print(f"Tokenizer loaded: {config.tokenizer_name}")
    except Exception as e:
        main_print(f"Error loading tokenizer: {e}")
        raise
    
    # ----------------------------------
    # Generation Parameters
    # ----------------------------------
    generation_params = {
        "temperature": getattr(config, 'generation_temperature', 1),
        "max_new_tokens": getattr(config, 'max_new_tokens', 1024),
        "do_sample": True,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    main_print(f"Generation parameters: {generation_params}")
    
    # ----------------------------------
    # Generate Synthetic Data
    # ----------------------------------
    main_print("\n=== GENERATING SYNTHETIC DATA ===")
    
    total_batches = len(dataloader)
    successful_batches = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Generating synthetic data")):
            try:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                
                # Store original inputs
                # unbind() splits batch tensor into individual sequences
                ds["input_ids"].extend(input_ids.cpu().unbind())
                ds["attention_mask"].extend(attention_mask.cpu().unbind())
                
                # Generate with teacher model
                generation_output = teacher_model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **generation_params
                )
                
                # Extract generated sequences
                generated_sequences = generation_output
                gen_only = generated_sequences[:, input_ids.shape[1]:]
                
                # Create labels for distillation
                # Labels should be -100 for original input (not trained on) and actual tokens for generated part
                max_len = generated_sequences.shape[1]
                labels = torch.full(
                    (input_ids.size(0), max_len), 
                    fill_value=-100, 
                    dtype=torch.long, 
                    device=generated_sequences.device
                )
                
                # Fill the generated part with actual tokens
                labels[:, input_ids.size(1):] = gen_only
                
                # Store labels (unbind to get individual sequences)
                ds["labels"].extend(labels.cpu().unbind())
                
                successful_batches += 1
                
                # Memory cleanup
                del generation_output, generated_sequences, gen_only, labels, input_ids, attention_mask
                torch.cuda.empty_cache()
                
            except Exception as e:
                main_print(f"Error processing batch {batch_idx}: {e}")
                # Continue with next batch instead of failing completely
                continue
    
    # ----------------------------------
    # Create and Save Dataset
    # ----------------------------------
    main_print(f"\nSuccessfully processed {successful_batches}/{total_batches} batches")
    main_print("Creating synthetic dataset...")
    
    try:
        # Create dataset from collected data
        synthetic_dataset = datasets.Dataset.from_dict(ds)
        synthetic_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        
        # Ensure output directory exists
        output_path = getattr(config, 'synthetic_dataset_path', 'synthetic_dataset')
        os.makedirs(output_path, exist_ok=True)
        
        # Save dataset
        synthetic_dataset.save_to_disk(output_path)
        main_print(f"Synthetic dataset saved to: {output_path}")
        main_print(f"Dataset contains {len(synthetic_dataset)} samples")
        
    except Exception as e:
        main_print(f"Error saving synthetic dataset: {e}")
        raise
    
    # ----------------------------------
    # Final Summary
    # ----------------------------------
    total_time = time.time() - start_time
    main_print(f"\nSynthetic dataset generation completed in {total_time/60:.2f} minutes")
    main_print(f"Generated {len(synthetic_dataset)} synthetic samples")
    
    return output_path














if __name__ == "__main__":
    try:
        output_path = generate_synthetic_dataset()
        main_print(f"\n✅ Synthetic dataset generation successful!")
        main_print(f"📁 Dataset saved to: {output_path}")
    except Exception as e:
        main_print(f"\n❌ Synthetic dataset generation failed: {e}")
        raise
