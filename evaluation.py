import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

def evaluate_model(model, test_dataloader, device='CPU', max_new_tokens=8, num_beams=4, no_repeat_ngram_size=4):
    model.eval()
    predicted_lyrics = []

    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Evaluating"):
            images = batch['image'].to(device)
            lyrics = batch['lyrics'].to(device)

            
            # Forward pass
            # outputs = model(pixel_values=images, labels=lyrics,
            #                  decoder_attention_mask=lyrics != model.config.pad_token_id)
            # logits = outputs.logits
            # predicted_tokens = logits.argmax(dim=-1)
            predicted_tokens = model.generate(images,
                                              max_new_tokens=max_new_tokens,
                                             num_beams=num_beams,
                                             no_repeat_ngram_size=no_repeat_ngram_size)
            
            # Convert predicted token IDs to text
            for i in range(len(predicted_tokens)):
                predicted_text = test_dataloader.dataset.tokenizer.decode(predicted_tokens[i])
                predicted_lyrics.append(predicted_text)
    
    true_lyrics = [test_dataloader.dataset.tokenizer.decode(item['lyrics']) for item in test_dataloader.dataset]

    for true, predicted in zip(true_lyrics, predicted_lyrics):
        print("True Lyrics:", true)
        print("Predicted Lyrics:", predicted)
        print("=" * 50)
       
    return true_lyrics, predicted_lyrics
