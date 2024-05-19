import numpy as np 
import pandas as pd
import csv
import nltk
import random
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import string
import torch
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer,GPT2Model, GPT2Config, AdamW
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu
from rouge_score import rouge_scorer
from bert_score import BERTScorer
from transformers import BertTokenizer, BertForMaskedLM, BertModel

# Function for text preprocessing
def preprocess_text(text):
    """
    Preprocesses the input text by converting to lowercase, removing punctuation and digits, tokenizing, and removing stop words.
    
    Args:
    text (str): Input text to be preprocessed.
    
    Returns:
    str: Preprocessed text.
    """
    # Convert to lowercase
    text = text.lower()

    # Remove punctuation and digits
    text = text.translate(str.maketrans('', '', string.punctuation + string.digits))

    # Tokenization
    tokens = word_tokenize(text)

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # Join the tokens back into a string
    preprocessed_text = ' '.join(tokens)

    return preprocessed_text

# Define a function to tokenize, convert text to indices, and pad sequences
def tokenize_and_pad(data_list, max_article_length=1021, max_highlights_length=1024):
    """
    Tokenizes the input data and pads sequences to specified lengths.
    
    Args:
    data_list (list): List of tuples containing articles and highlights.
    max_article_length (int): Maximum length of articles.
    max_highlights_length (int): Maximum length of highlights.
    
    Returns:
    list: Tokenized and padded data list.
    """
    tokenized_data_list = []
    for article, highlights in data_list:
        # Tokenize and convert to indices
        article_tokens = tokenizer.encode(article, add_special_tokens=True)
        highlights_tokens = tokenizer.encode(highlights, add_special_tokens=True)

        # Pad sequences to specified lengths
        padded_article_tokens = torch.tensor(article_tokens + [tokenizer.convert_tokens_to_ids(pad_token)] * (max_article_length - len(article_tokens)))
        padded_highlights_tokens = torch.tensor(highlights_tokens + [tokenizer.convert_tokens_to_ids(pad_token)] * (max_highlights_length - len(highlights_tokens)))

        # Append to the tokenized_data_list only if both token lists are not empty
        if len(article_tokens) > 0 and len(highlights_tokens) > 0:
            tokenized_data_list.append((padded_article_tokens, padded_highlights_tokens))

    return tokenized_data_list

def calculate_bleu_score(machine_results, reference_texts):
    bleu_score = corpus_bleu([[ref.split()] for ref in reference_texts], [gen.split() for gen in machine_results])
    return bleu_score

def calculate_rouge_scores(generated_answers, ground_truth):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    total_rouge1, total_rouge2, total_rougeL = 0, 0, 0
    for gen, ref in zip(generated_answers, ground_truth):
        scores = scorer.score(gen, ref)
        total_rouge1 += scores['rouge1'].fmeasure
        total_rouge2 += scores['rouge2'].fmeasure
        total_rougeL += scores['rougeL'].fmeasure
    average_rouge1 = total_rouge1 / len(generated_answers)
    average_rouge2 = total_rouge2 / len(generated_answers)
    average_rougeL = total_rougeL / len(generated_answers)
    return average_rouge1, average_rouge2, average_rougeL

def calculate_bert_score(generated_answers, ground_truth):
    scorer = BERTScorer(model_type='bert-base-uncased')
    P, R, F1 = scorer.score(generated_answers, ground_truth)
    avg_precision = sum(p.mean() for p in P) / len(P)
    avg_recall = sum(r.mean() for r in R) / len(R)
    avg_f1 = sum(f1.mean() for f1 in F1) / len(F1)
    return avg_precision, avg_recall, avg_f1



csv_file_path = 'test.csv'

# List to store tuples of (article, highlights)
test_data_list = []

# Read CSV file and extract relevant columns
# Read CSV file and extract relevant columns
with open(csv_file_path, 'r', encoding='utf-8') as file:
    csv_reader = csv.DictReader(file)
    for row in csv_reader:
        article = row.get('article', '')
        highlights = row.get('highlights', '')
        test_data_list.append((article, highlights))
        


csv_file_path = 'train.csv'

# List to store tuples of (article, highlights)
train_data_list = []

# Read CSV file and extract relevant columns
with open(csv_file_path, 'r', encoding='utf-8') as file:
    csv_reader = csv.DictReader(file)
    for row in csv_reader:
        article = row.get('article', '')
        highlights = row.get('highlights', '')
        train_data_list.append((article, highlights))
        
        


csv_file_path = 'validation.csv'

# List to store tuples of (article, highlights)
val_data_list = []

# Read CSV file and extract relevant columns
with open(csv_file_path, 'r', encoding='utf-8') as file:
    csv_reader = csv.DictReader(file)
    for row in csv_reader:
        article = row.get('article', '')
        highlights = row.get('highlights', '')
        val_data_list.append((article, highlights))



# Set the seed for reproducibility
random.seed(14)  # You can choose any seed value

# Calculate 1% of the original data size
one_percent_size = int(0.01 * len(test_data_list))

# Randomly sample 1% of the data
onetest_data_list = random.sample(test_data_list, one_percent_size)
# Calculate 1% of the original data size
one_percent_size = int(0.01 * len(train_data_list))

# Randomly sample 1% of the data
onetrain_data_list = random.sample(train_data_list, one_percent_size)
# Calculate 1% of the original data size
one_percent_size = int(0.01 * len(val_data_list))

# Randomly sample 1% of the data
oneval_data_list = random.sample(val_data_list, one_percent_size)

# Apply preprocessing to your data
ptrain_data_list = [(preprocess_text(article), preprocess_text(highlights)) for article, highlights in onetrain_data_list]
ptest_data_list = [(preprocess_text(article), preprocess_text(highlights)) for article, highlights in onetest_data_list]
pval_data_list = [(preprocess_text(article), preprocess_text(highlights)) for article, highlights in oneval_data_list]


# Load GPT-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Define a pad token and add it to the tokenizer
pad_token = tokenizer.eos_token
tokenizer.add_tokens([pad_token])

# Apply tokenization and padding to your datasets
max_article_length = 1021
max_highlights_length = 1024
tokenized_train_data_list = tokenize_and_pad(ptrain_data_list, max_article_length, max_highlights_length)
tokenized_test_data_list = tokenize_and_pad(ptest_data_list, max_article_length, max_highlights_length)
tokenized_val_data_list = tokenize_and_pad(pval_data_list, max_article_length, max_highlights_length)

# Initialize lists to store input and target ids
input_ids_train = []
target_ids_train = []

# Specify maximum lengths
max_article_length = 1021
max_highlights_length = 1024

# Iterate through the tokenized data
for padded_article_tokens, padded_highlights_tokens in tokenized_train_data_list:
    # Truncate article tokens if greater than max_article_length
    truncated_article_tokens = padded_article_tokens[:max_article_length]

    # Truncate highlights tokens if greater than max_highlights_length
    truncated_highlights_tokens = padded_highlights_tokens[:max_highlights_length]

    # Append truncated article tokens to input_ids_train
    input_ids_train.append(truncated_article_tokens)

    # Append truncated highlights tokens to target_ids_train
    target_ids_train.append(truncated_highlights_tokens)

# Convert the lists to PyTorch tensors
input_ids_train = torch.stack(input_ids_train)
target_ids_train = torch.stack(target_ids_train)





# Initialize lists to store input and target ids
input_ids_val = []
target_ids_val = []

# Specify maximum lengths
max_article_length = 1021
max_highlights_length = 1024

# Iterate through the tokenized data
for padded_article_tokens, padded_highlights_tokens in tokenized_val_data_list:
    # Truncate article tokens if greater than max_article_length
    truncated_article_tokens = padded_article_tokens[:max_article_length]

    # Truncate highlights tokens if greater than max_highlights_length
    truncated_highlights_tokens = padded_highlights_tokens[:max_highlights_length]

    # Append truncated article tokens to input_ids_train
    input_ids_val.append(truncated_article_tokens)

    # Append truncated highlights tokens to target_ids_train
    target_ids_val.append(truncated_highlights_tokens)

# Convert the lists to PyTorch tensors
input_ids_val = torch.stack(input_ids_val)
target_ids_val = torch.stack(target_ids_val)


# Load GPT-2 model and tokenizer
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
gpt2_model = GPT2LMHeadModel.from_pretrained(model_name)

# Define the number of prompts and embedding size
num_prompts_token = 3  # "summarize the following text"
embedding_size = 768

# Define a specific sentence
sentence = "summarize"

# Tokenize the sentence
input_ids = tokenizer.encode(sentence, return_tensors='pt')

# Get the embeddings for the input_ids from the GPT-2 model
gpt2_embeddings = gpt2_model.transformer.wte(input_ids)

# Create an embedding layer for soft prompts and initialize with the sentence embeddings
soft_prompt_embeddings = nn.Embedding(num_prompts_token, embedding_size)
soft_prompt_embeddings.weight.data.copy_(gpt2_embeddings.squeeze(0))
# print("Shape of soft prompt embeddings:", soft_prompt_embeddings.weight.data.shape)

# Concatenate soft prompt embeddings at the beginning of the input sequence
class GPT2WithPromptTuning(nn.Module):
    def __init__(self, gpt2_model, soft_prompt_embeddings):
        super(GPT2WithPromptTuning, self).__init__()
        self.gpt2_model = gpt2_model
        self.soft_prompt_embeddings = soft_prompt_embeddings
    
    def forward(self, input_ids, soft_prompt_ids):
        # Get the embeddings for the input_ids from the GPT-2 model
        gpt2_embeddings = self.gpt2_model.transformer.wte(input_ids)
        # Get the embeddings for the soft prompts
        soft_prompt_embeds = self.soft_prompt_embeddings(soft_prompt_ids)
         # Concatenate the embeddings
        # print("Shape of soft prompt embeddings:", soft_prompt_embeds.shape)
        # print("soft prompt embeddings:", soft_prompt_embeds)

        # print("Shape of input embeddings:", gpt2_embeddings.shape)
        # print("soft input prompt embeddings:", gpt2_embeddings)
        
        # Concatenate the embeddings
        embeddings = torch.cat([soft_prompt_embeds, gpt2_embeddings], dim=0)

        # print("Shape of concatenated embeddings:", embeddings.shape)
        # print("soft of concatenated embeddings:", embeddings)
        
        # Pass the concatenated embeddings through the GPT-2 model
        outputs = self.gpt2_model(inputs_embeds=embeddings)
        
        return outputs


# Initialize the model
model = GPT2WithPromptTuning(gpt2_model, soft_prompt_embeddings)

# Freeze GPT-2 model weights
for param in model.gpt2_model.parameters():
    param.requires_grad = False

# Define hyperparameters
batch_size = 8
epochs = 2
learning_rate = 2e-3
gradient_clip_value = 1.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move model to GPU
model.to(device)

# Define optimizer and criterion
optimizer = torch.optim.AdamW(model.soft_prompt_embeddings.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss(ignore_index=-100)

soft_prompt_ids = torch.tensor([0, 1, 2])



# Lists to store scores
train_bleu_scores = []
train_bert_scores = []
train_rouge1_scores = []
train_rouge2_scores = []
train_rougeL_scores = []

val_bleu_scores = []
val_bert_scores = []
val_rouge1_scores = []
val_rouge2_scores = []
val_rougeL_scores = []


# Training loop
for epoch in range(epochs):
    # Create a tqdm progress bar for the training data
    data_iterator = tqdm(zip(input_ids_train, target_ids_train), desc=f'Epoch {epoch + 1}', total=len(input_ids_train))
    
    for input_ids, target_ids in data_iterator:
        optimizer.zero_grad()

        # Move input and target tensors to GPU
        input_ids, target_ids = input_ids.to(device), target_ids.to(device)
        # print("input",input_ids.shape,"target",target_ids.shape)
        # print("input",len(input_ids),"target",len(target_ids))
        
        # Assuming you have a soft_prompt_ids for each training instance
        # If not, you might need to modify this part accordingly
        outputs = model(input_ids, soft_prompt_ids.to(device))
        logits = outputs.logits if hasattr(outputs, "logits") else outputs.last_hidden_state

        loss = criterion(logits, target_ids)
        loss.backward()

        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_value)

        optimizer.step()

        # Update the progress bar description with the current loss
        data_iterator.set_postfix(loss=loss.item())

        # Convert tensor predictions and references to lists
        predictions = logits.argmax(dim=-1).squeeze(0).tolist()
        references = target_ids.squeeze(0).tolist()

        # Calculate BLEU Score for training
        bleu_score = calculate_bleu_score([tokenizer.decode(predictions)], [tokenizer.decode(references)])
        train_bleu_scores.append(bleu_score)

        # Calculate BERTScore for training
        bert_precision, bert_recall, bert_f1 = calculate_bert_score([tokenizer.decode(predictions)], [tokenizer.decode(references)])
        train_bert_scores.append(bert_f1)

        # Calculate ROUGE Scores for training
        rouge1, rouge2, rougeL = calculate_rouge_scores([tokenizer.decode(predictions)], [tokenizer.decode(references)])
        train_rouge1_scores.append(rouge1)
        train_rouge2_scores.append(rouge2)
        train_rougeL_scores.append(rougeL)


#     # Validation loop
    model.eval()
    val_losses = []
    val_bleu_scores_epoch = []
    val_bert_scores_epoch = []
    val_rouge1_scores_epoch = []
    val_rouge2_scores_epoch = []
    val_rougeL_scores_epoch = []
    val_losses = []
    with torch.no_grad():
        for input_ids_val, target_ids_val in zip(input_ids_val, target_ids_val):
            input_ids_val, target_ids_val = input_ids_val.to(device), target_ids_val.to(device)
            outputs_val = model(input_ids_val, soft_prompt_ids.to(device))
            logits_val = outputs_val.logits if hasattr(outputs_val, "logits") else outputs_val.last_hidden_state
            loss_val = criterion(logits_val, target_ids_val)
            val_losses.append(loss_val.item())
            # Convert tensor predictions and references to lists
            predictions_val = logits_val.argmax(dim=-1).squeeze(0).tolist()
            references_val = target_ids_val.squeeze(0).tolist()

            # Calculate BLEU Score for validation
            bleu_score_val = calculate_bleu_score([tokenizer.decode(predictions_val)], [tokenizer.decode(references_val)])
            val_bleu_scores_epoch.append(bleu_score_val)

            # Calculate BERTScore for validation
            bert_precision_val, bert_recall_val, bert_f1_val = calculate_bert_score([tokenizer.decode(predictions_val)], [tokenizer.decode(references_val)])
            val_bert_scores_epoch.append(bert_f1_val)

            # Calculate ROUGE Scores for validation
            rouge1_val, rouge2_val, rougeL_val = calculate_rouge_scores([tokenizer.decode(predictions_val)], [tokenizer.decode(references_val)])
            val_rouge1_scores_epoch.append(rouge1_val)
            val_rouge2_scores_epoch.append(rouge2_val)
            val_rougeL_scores_epoch.append(rougeL_val)

    # Calculate average validation loss
    avg_val_loss = sum(val_losses) / len(val_losses)
    print("epoch :", epoch + 1,"train_loss :", loss.item(),"val_loss :", avg_val_loss)

    # Calculate average validation scores
    avg_bleu_score_val = sum(val_bleu_scores_epoch) / len(val_bleu_scores_epoch)
    avg_bert_score_val = sum(val_bert_scores_epoch) / len(val_bert_scores_epoch)
    avg_rouge1_score_val = sum(val_rouge1_scores_epoch) / len(val_rouge1_scores_epoch)
    avg_rouge2_score_val = sum(val_rouge2_scores_epoch) / len(val_rouge2_scores_epoch)
    avg_rougeL_score_val = sum(val_rougeL_scores_epoch) / len(val_rougeL_scores_epoch)

    print("Validation BLEU Score:", avg_bleu_score_val)
    print("Validation BERTScore:", avg_bert_score_val)
    print("Validation ROUGE-1 Score:", avg_rouge1_score_val)
    print("Validation ROUGE-2 Score:", avg_rouge2_score_val)
    print("Validation ROUGE-L Score:", avg_rougeL_score_val)

    # Append validation scores
    val_bleu_scores.append(avg_bleu_score_val)
    val_bert_scores.append(avg_bert_score_val)
    val_rouge1_scores.append(avg_rouge1_score_val)
    val_rouge2_scores.append(avg_rouge2_score_val)
    val_rougeL_scores.append(avg_rougeL_score_val)

    # Set the model back to training mode
    model.train()

# Close the tqdm progress bar
data_iterator.close()


# Calculate average scores for training
avg_train_bleu_score = sum(train_bleu_scores) / len(train_bleu_scores)
avg_train_bert_score = sum(train_bert_scores) / len(train_bert_scores)
avg_train_rouge1_score = sum(train_rouge1_scores) / len(train_rouge1_scores)
avg_train_rouge2_score = sum(train_rouge2_scores) / len(train_rouge2_scores)
avg_train_rougeL_score = sum(train_rougeL_scores) / len(train_rougeL_scores)

print("Average Training BLEU Score:", avg_train_bleu_score)
print("Average Training BERTScore:", avg_train_bert_score)
print("Average Training ROUGE-1 Score:", avg_train_rouge1_score)
print("Average Training ROUGE-2 Score:", avg_train_rouge2_score)
print("Average Training ROUGE-L Score:", avg_train_rougeL_score)

# Calculate average scores for validation
avg_val_bleu_score = sum(val_bleu_scores) / len(val_bleu_scores)
avg_val_bert_score = sum(val_bert_scores) / len(val_bert_scores)
avg_val_rouge1_score = sum(val_rouge1_scores) / len(val_rouge1_scores)
avg_val_rouge2_score = sum(val_rouge2_scores) / len(val_rouge2_scores)
avg_val_rougeL_score = sum(val_rougeL_scores) / len(val_rougeL_scores)

print("Average Validation BLEU Score:", avg_val_bleu_score)
print("Average Validation BERTScore:", avg_val_bert_score)
print("Average Validation ROUGE-1 Score:", avg_val_rouge1_score)
print("Average Validation ROUGE-2 Score:", avg_val_rouge2_score)
print("Average Validation ROUGE-L Score:", avg_val_rougeL_score)


 # Save model weights and tokenizer
torch.save(model.state_dict(), 'Summarize_weights1.pth')

# model.save_pretrained("Summarize", safe_serialization=True)
# tokenizer.save_pretrained("Summarizetoken")


