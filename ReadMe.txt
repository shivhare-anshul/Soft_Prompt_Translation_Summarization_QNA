1. Overview
   This repository contains code and documentation for three natural language processing tasks: summarization, question answering, and machine   translation.

2. Tasks and Files

   For each task, both .py and .ipynb files are provided.

   Summarization
     File: summarize.py
     To run training: python summarize.py

   Question Answering
     File: QNA.py
     To run training: python QNA.py

   Machine Translation
     File: translation.py
     To run training: python translation.py

The .py files are in separate folders; each folder corresponds to a specific task. Additionally, it's important to note that in order to run the .py files successfully, you must place the CSV and JSON files in their respective task folders.
**Note:** To run these files, you are required to be in a dedicated Python environment. The required libraries are listed below and also available in the requirement.txt file.


!pip install nltk 
!pip install transformer
!pip install pandas
!pip install torch
!pip install tqdm
!pip install scikit-learn
!pip install rouge-score
!pip install bert-score
!pip install numpy

3. Theory and Architecture

   Refer to the "Theory" document for discussions on soft prompts, tuning process, architecture, and a comparison with hard prompts (Soft prompt vs Hard prompt).


4. Evaluation and Results


  The "Evaluation and Results Report" document contains:
     1. Hyperparameters used for training.
     2. Methodology employed.
     3. Dataset descriptions and tokens utilized.
     4. Performance evaluation with discussion on results for all three tasks.