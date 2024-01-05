# Hindi eval

This is a repository that hosts the recipes, scripts and datasets for evaluation of Hindi LLMs

## Directory structure

|folder|description|
|--|--|
|data|Hosts public datasets also available on Huggingface|
|docs|Docs and guides for running recipes|
|notebooks|Notebooks for experimentation and cleaning adhoc data|
|src|Source code (recipes, scripts to generate dataset and scripts to evaluate)|

## Eval datasets

We want to evaluate LLMs on these tasks

### Translation

- We want to evaluate the performance of LLM on Hindi to English and English to Hindi translations
- Steps followed to evaluate translation performance
    1. Select an **audit model** (ex: Llama 7B). Evaluate it's performance on a specific task dataset (eg: MMLU)
    2. Select a **candidate model** (ex: OpenHathi)
    3. Use the candidate mode to translate the specific task dataset to Hindi and back to English
    4. Evaluate audit model's performance on the newly translated dataset (in step 3)
    5. **Measurement**: Whatever the drop of performance is for the audit model, can be measured as the translation quality of the candidate model. Ex: If Llama 7B scores 40% on MMLU dataset, but on the translated MMLU dataset (Hi->En then En->Hi using candidate model) it achieves 30%, the 10% drop can be attributed to the loss in translation

### Problem solving and subject expertise

- We want to evaluate LLM's performance in solving logical questions and subject specific questions in Hindi
- For this, we generate synthetic data using prompting. We select 4 subjects - Physics, Maths, Chemistry and Biology
- The synthetic data contains MCQ question and answer pair that can be evaluated against any LLM
- **Measurement**: Accuracy on the dataset

### Retrieval

- We want to evaluate LLM's retreival performance
- For this, we scrape latest Hindi news articles and generate synthetic question/answer pairs using GPT. The questions are of type: Fill in the blanks, True False, and MCQ
- The synthetic dataset consists of MCQ based on the passage that can be evaluated against any LLM
- **Measurement**: Accuracy on the dataset

- summarize fictional chapters/stories and create questions using cgpt4

### Tasks in Hindi

TBA

- Summarize a given text (English, Hindi, Romanized Hindi) in Hindi
- Write in a style of a known personality in Hindi
