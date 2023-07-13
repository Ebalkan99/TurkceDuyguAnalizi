import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import get_linear_schedule_with_warmup

# Örnek veri setini yükleyin veya kendi veri setinizi kullanın
train_texts = ["Bu film harika!", "Bu kitap sıkıcı."]
train_labels = [1, 0]

# BERT tokenizer'ı yükleyin
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Verileri BERT için giriş formatına dönüştürün
train_encodings = tokenizer(train_texts, truncation=True, padding=True)

# Veri kümesini oluşturun
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = MyDataset(train_encodings, train_labels)

# Modeli yükleyin veya eğitin
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Eğitim parametrelerini ayarlayın
batch_size = 8
epochs = 5
lr = 2e-5

# DataLoader'ı oluşturun
train_sampler = RandomSampler(train_dataset)
train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size)

# Optimizer'ı ve scheduler'ı tanımlayın
optimizer = AdamW(model.parameters(), lr=lr)
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# Modeli eğitin
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

for epoch in range(epochs):
    model.train()
    total_loss = 0
    
    for batch in train_dataloader:
        optimizer.zero_grad()
        
        inputs = {key: val.to(device) for key, val in batch.items()}
        outputs = model(**inputs)
        
        loss = outputs.loss
        total_loss += loss.item()
        
        loss.backward()
        optimizer.step()
        scheduler.step()
    
    average_loss = total_loss / len(train_dataloader)
    print(f'Epoch {epoch+1}/{epochs} - Average Loss: {average_loss}')

# Eğitilmiş modeli kaydedin
model.save_pretrained('path/to/save/model')
tokenizer.save_pretrained('path/to/save/tokenizer')
