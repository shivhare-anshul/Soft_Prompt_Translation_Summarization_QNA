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
import json
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Model, GPT2Config, AdamW
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu
from rouge_score import rouge_scorer
from bert_score import BERTScorer
from transformers import BertTokenizer, BertForMaskedLM, BertModel

def squad_json_to_dataframe(file_path, record_path=['data','paragraphs','qas','answers']):
    """
    Convert SQuAD JSON file to pandas DataFrame.

    Args:
    file_path (str): Path to the SQuAD JSON file.
    record_path (list, optional): Path to the records in the JSON file. Defaults to ['data','paragraphs','qas','answers'].

    Returns:
    pandas.DataFrame: DataFrame containing SQuAD data.
    """
    file = json.loads(open(file_path).read())
    # parsing different level's in the json file
    js = pd.json_normalize(file, record_path)
    m = pd.json_normalize(file, record_path[:-1])
    r = pd.json_normalize(file, record_path[:-2])
    # combining it into single dataframe
    idx = np.repeat(r['context'].values, r.qas.str.len())
    m['context'] = idx
    data = m[['id','question','context','answers']].set_index('id').reset_index()
    data['c_id'] = data['context'].factorize()[0]
    return data

def preprocess_text_en(data):
    """
    Preprocess English text data.

    Args:
    data (list): List of text data.

    Returns:
    list: Preprocessed text data.
    """
    data2 = []
    for text in data:
        # Skip None values
        if text is None:
            continue
        text = text.lower()

        # Tokenization
        tokens = word_tokenize(text)

        # Remove stop words
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]

        # Join the tokens back into a string
        preprocessed_text = ' '.join(tokens)
        data2.append(preprocessed_text)

    return data2

def calculate_bleu_score(machine_results, reference_texts):
    """
    Calculate BLEU score.

    Args:
    machine_results (list): List of machine-generated texts.
    reference_texts (list): List of reference texts.

    Returns:
    float: BLEU score.
    """
    bleu_score = corpus_bleu([[ref.split()] for ref in reference_texts], [gen.split() for gen in machine_results])
    return bleu_score

def calculate_rouge_scores(generated_answers, ground_truth):
    """
    Calculate ROUGE scores.

    Args:
    generated_answers (list): List of generated answers.
    ground_truth (list): List of ground truth answers.

    Returns:
    tuple: Average ROUGE-1, ROUGE-2, and ROUGE-L scores.
    """
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
    """
    Calculate BERTScore.

    Args:
    generated_answers (list): List of generated answers.
    ground_truth (list): List of ground truth answers.

    Returns:
    tuple: Average precision, recall, and F1 scores.
    """
    scorer = BERTScorer(model_type='bert-base-uncased')
    P, R, F1 = scorer.score(generated_answers, ground_truth)
    avg_precision = sum(p.mean() for p in P) / len(P)
    avg_recall = sum(r.mean() for r in R) / len(R)
    avg_f1 = sum(f1.mean() for f1 in F1) / len(F1)
    return avg_precision, avg_recall, avg_f1

# Define a function to tokenize, convert text to indices, and pad sequences
def tokenize_and_pad(data_list, max_question_length=1018, max_answer_length=1024):
    """
    Tokenize, convert text to indices, and pad sequences.

    Args:
    data_list (list): List of tuples containing question and answer texts.
    max_question_length (int, optional): Maximum length of question sequence. Defaults to 1018.
    max_answer_length (int, optional): Maximum length of answer sequence. Defaults to 1024.

    Returns:
    list: Tokenized and padded data list.
    """
    tokenized_data_list = []
    for question, answer in data_list:
        # Tokenize and convert to indices
        question_tokens = tokenizer.encode(question, add_special_tokens=True)
        answer_tokens = tokenizer.encode(answer, add_special_tokens=True)

        # Pad sequences to specified lengths
        padded_question_tokens = torch.tensor(question_tokens + [tokenizer.convert_tokens_to_ids(pad_token)] * (max_question_length - len(question_tokens)))
        padded_answer_tokens = torch.tensor(answer_tokens + [tokenizer.convert_tokens_to_ids(pad_token)] * (max_answer_length - len(answer_tokens)))

        # Append to the tokenized_data_list only if both token lists are not empty
        if len(question_tokens) > 0 and len(answer_tokens) > 0:
            tokenized_data_list.append((padded_question_tokens, padded_answer_tokens))

    return tokenized_data_list

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

train = 'train-v2.0.json'
dev = 'dev-v2.0.json'
train_df = squad_json_to_dataframe(train)
dev_df = squad_json_to_dataframe(dev)

train_df['question_context'] = 'Question: ' + train_df['question'] + ' Context: ' + train_df['context']
dev_df['question_context'] = 'Question: ' + dev_df['question'] + ' Context: ' + dev_df['context']

# Extract 'text' value from 'answers' column for each row
train_df['answers'] = train_df['answers'].apply(lambda x: x[0]['text'] if len(x) > 0 else None)
dev_df['answers'] = dev_df['answers'].apply(lambda x: x[0]['text'] if len(x) > 0 else None)

# Extract 'question_context' column into a list
question_context_list_train = train_df['question_context'].tolist()
# Extract 'answers' column into a list
answers_list_train = train_df['answers'].tolist()

# Extract 'question_context' column into a list
question_context_list_dev = dev_df['question_context'].tolist()
# Extract 'answers' column into a list
answers_list_dev = dev_df['answers'].tolist()

question_context_list_train = question_context_list_train[:5000]
answers_list_train = answers_list_train[:5000]

question_context_list_dev = question_context_list_dev[:1000]
answers_list_dev = answers_list_dev[:1000]

question_context_list_train = preprocess_text_en(question_context_list_train)
answers_list_train = preprocess_text_en(answers_list_train)

question_context_list_dev = preprocess_text_en(question_context_list_dev)
answers_list_dev = preprocess_text_en(answers_list_dev)

mapped_list_1 = list(zip(question_context_list_train, answers_list_train))
mapped_list_2 = list(zip(question_context_list_dev, answers_list_dev))

# Load GPT-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Define a pad token and add it to the tokenizer
pad_token = tokenizer.eos_token
tokenizer.add_tokens([pad_token])

tokenized_data_train_list = tokenize_and_pad(mapped_list_1, 1018, 1024)
tokenized_data_val_list = tokenize_and_pad(mapped_list_2, 1018, 1024)

# Initialize the model
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
gpt2_model = GPT2LMHeadModel.from_pretrained(model_name)

# Define the number of prompts and embedding size
num_prompts = 6  # "summarize the following text"
embedding_size = 768

# Define a specific sentence
sentence = "Answer the question using given context"

# Tokenize the sentence
input_ids = tokenizer.encode(sentence, return_tensors='pt')

# Get the embeddings for the input_ids from the GPT-2 model
gpt2_embeddings = gpt2_model.transformer.wte(input_ids)

# Create an embedding layer for soft prompts and initialize with the sentence embeddings
soft_prompt_embeddings = nn.Embedding(num_prompts, embedding_size)
soft_prompt_embeddings.weight.data.copy_(gpt2_embeddings.squeeze(0))

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
        embeddings = torch.cat([soft_prompt_embeds, gpt2_embeddings], dim=0)
        
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

# Define optimizer and criterion
optimizer = torch.optim.AdamW(model.soft_prompt_embeddings.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss(ignore_index=-100)

soft_prompt_ids = torch.tensor([0, 1, 2, 3, 4, 5])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move model to GPU
model.to(device)

# Training loop
for epoch in range(epochs):
    # Create a tqdm progress bar for the training data
    data_iterator = tqdm(zip(input_ids_train, target_ids_train), desc=f'Epoch {epoch + 1}', total=len(input_ids_train))
    
    for input_ids, target_ids in data_iterator:
        optimizer.zero_grad()

        # Move input and target tensors to GPU
        input_ids, target_ids = input_ids.to(device), target_ids.to(device)
        
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

    # Validation loop
    model.eval()
    val_losses = []
    val_bleu_scores_epoch = []
    val_bert_scores_epoch = []
    val_rouge1_scores_epoch = []
    val_rouge2_scores_epoch = []
    val_rougeL_scores_epoch = []
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

# Save model weights
torch.save(model.state_dict(), 'QNA.pth')
# model.save_pretrained("QNA", safe_serialization=True)
# tokenizer.save_pretrained("QNAtoken")
