import torch
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from diffusion_core.model import DiffusionTransformer
from diffusion_core.data import SimpleTextDataset, DiffusionCollator
from diffusion_core.loss import diffusion_cross_entropy
from diffusion_core.inference import DiffusionSampler

def main():
    # 1. Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    # Add a special MASK token if it doesn't exist
    if "[MASK]" not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": ["[MASK]"]})
    mask_token_id = tokenizer.convert_tokens_to_ids("[MASK]")

    # 2. Data
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence is transforming the world.",
        "Diffusion models are a class of generative models.",
    ]
    tokenized_data = [tokenizer.encode(t) for t in texts]
    dataset = SimpleTextDataset(tokenized_data)
    collator = DiffusionCollator(tokenizer, mask_token_id, max_length=32)
    loader = DataLoader(dataset, batch_size=2, collate_fn=collator)

    # 3. Model
    model = DiffusionTransformer(
        vocab_size=len(tokenizer),
        max_position_embeddings=128,
        hidden_size=256,
        num_layers=4,
        num_heads=4,
        intermediate_size=1024
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # 4. Simple Training Loop
    model.train()
    for epoch in range(5):
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            target_ids = batch["target_ids"].to(device)
            p_mask_scalar = batch["p_mask_scalar"].to(device)

            logits = model(input_ids, attention_mask)
            loss = diffusion_cross_entropy(logits, target_ids, p_mask_scalar)
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    # 5. Inference
    sampler = DiffusionSampler(model, tokenizer, mask_token_id, device)
    prompt = "The quick brown"
    prompt_ids = tokenizer.encode(prompt)
    
    print(f"\nPrompt: {prompt}")
    generated_ids = sampler.generate(
        prompt_ids, 
        max_new_tokens=5, 
        block_size=5, 
        steps_per_block=10
    )
    output_text = tokenizer.decode(generated_ids)
    print(f"Generated: {output_text}")

if __name__ == "__main__":
    main()
