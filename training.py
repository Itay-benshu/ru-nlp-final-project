import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

def train_model(model, train_dataloader, validation_dataloader, num_epochs=5, learning_rate=0.001, device='CPU'):
    model = model.to(device)
    optimized_parameters = []
    for param_name, param in model.named_parameters():
        if 'crossattention' in param_name or 'cross_attn' in param_name or (param_name in 
        ['decoder.transformer.wte.weight', 'decoder.transformer.wpe.weight']):
            optimized_parameters.append(param)
            
    optimizer = optim.Adam(optimized_parameters, lr=learning_rate)
#     optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    train_losses = []
    val_losses = []
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0


        progress_bar = tqdm(enumerate(train_dataloader), 
                            total=len(train_dataloader))

        predicted_lyrics = []
        for batch_idx, batch in progress_bar:
            optimizer.zero_grad()

            images = batch['image'].to(device)
            lyrics = batch['lyrics'].to(device)

            # Forward pass
            outputs = model(pixel_values=images, labels=lyrics, 
                            decoder_attention_mask=lyrics != model.config.pad_token_id)
            loss = outputs.loss

            # Backpropagation and optimization step
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # Training generation
            
            # predicted_tokens = model.generate(images, max_new_tokens=16)

            # # Convert predicted token IDs to text
            # for i in range(len(predicted_tokens)):
            #     predicted_text = train_dataloader.dataset.tokenizer.decode(predicted_tokens[i])
            #     predicted_lyrics.append(predicted_text)
            
            progress_bar.set_description(f"Epoch {epoch+1}/{num_epochs} - Loss: {loss.item():.4f}")
        
        # true_lyrics = [train_dataloader.dataset.tokenizer.decode(item['lyrics']) for item in train_dataloader.dataset]
        # print('Training generation:')
        # for true, predicted in zip(true_lyrics, predicted_lyrics):
        #     print("True Lyrics:", true.strip())
        #     print("Predicted Lyrics:", predicted.strip())
        #     print("=" * 50)

        average_loss = total_loss / len(train_dataloader)
        train_losses.append(average_loss)
        print(f"Epoch {epoch+1}/{num_epochs} - Average Loss: {average_loss:.4f}")

        # Validation
        model.eval()
        total_val_loss = 0.0
        predicted_lyrics = []

        val_progress_bar = tqdm(validation_dataloader, total=len(validation_dataloader), leave=False)
        with torch.no_grad():
            for val_batch in val_progress_bar:
                val_images = val_batch['image'].to(device)
                val_lyrics = val_batch['lyrics'].to(device)

                val_outputs = model(pixel_values=val_images, labels=val_lyrics,
                                     decoder_attention_mask=val_lyrics != model.config.pad_token_id)
                val_logits = val_outputs.logits

                val_loss = val_outputs.loss
                total_val_loss += val_loss.item()
                
                val_progress_bar.set_description(f"Epoch {epoch+1}/{num_epochs} - Validation Loss: {val_loss.item():.4f}")

                # predicted_tokens = model.generate(val_images, max_new_tokens=16)

                # # Convert predicted token IDs to text
                # for i in range(len(predicted_tokens)):
                #     predicted_text = validation_dataloader.dataset.tokenizer.decode(predicted_tokens[i])
                #     predicted_lyrics.append(predicted_text)
        
        # true_lyrics = [validation_dataloader.dataset.tokenizer.decode(item['lyrics']) for item in validation_dataloader.dataset]
        
        # print('Validation generation:')
        # for true, predicted in zip(true_lyrics, predicted_lyrics):
        #     print("True Lyrics:", true.strip())
        #     print("Predicted Lyrics:", predicted.strip())
        #     print("=" * 50)

        average_val_loss = total_val_loss / len(validation_dataloader)
        val_losses.append(average_val_loss)
        print(f"Epoch {epoch+1}/{num_epochs} - Average Validation Loss: {average_val_loss:.4f}")
        
    return train_losses, val_losses