# Eval dataset

## Evaluation

We want to publish a report on following tests which can evaluate any model for Hindi. These are the use cases of a chat bot on which we will evaluate.

### Translation

- Hindi to English and English to Hindi
- We want to create a new dataset on which the model has not been trained
- Take a dataset which is not available in Hindi eg: Wikipedia english datasets, Journals, Articles etc.
- **Task**: Translate using model
- **Measurement**: Perplexity, GPT4, Reverse translation comparison

### Calculations in Hindi

- Open benchmark datasets + NCERT question papers (latest)
- Synthetic data generation
- **Task**: MCQ on one word answers to mathematical questions
- **Measurement**: Accuracy

### Retrieval from context

- Synthetic data generation and asking to summarize, answer questions on the passage and generate more content related to that
- Get journals, articles etc. published after Llama 2 release data, convert to hindi
- **Task**: Summarize, answer based on passage
- **Measurement**: GPT4

### Tasks in Hindi

TBA
