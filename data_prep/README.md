# Eval dataset

## Evaluation

We want to publish a report on following tests which can evaluate any model for Hindi. These are the use cases of a chat bot on which we will evaluate.

### Translation

- Hindi to English and English to Hindi
- We want to create a new dataset on which the model has not been trained
- Take a dataset which is not available in Hindi eg: Wikipedia english datasets, Journals, Articles etc.
- **Task**: Translate using model
- **Measurement**: Perplexity, GPT4, Reverse translation comparison

- evaluate llama's performance on mmlu
- translate using open hathi
- translate back using gpt-4
- evaluate llama's performance on translated mmlu
- translate 10 articles back and forth using open hathi
- measure 1: performance drop, perplexity in hindi language
- exact translation error from gpt 4

### Calculations in Hindi

- Open benchmark datasets + NCERT question papers (latest)
- Synthetic data generation
- **Task**: MCQ on one word answers to mathematical questions
- **Measurement**: Accuracy

- use gpt 4 to create math questions in hindi from all chapters in maths
- general mental maths questions from open papers

### Retrieval from context

- Synthetic data generation and asking to summarize, answer questions on the passage and generate more content related to that
- Get journals, articles etc. published after Llama 2 release data, convert to hindi
- **Task**: Answer based on passage
- **Measurement**: GPT4

- summarize fictional chapters/stories and create questions using cgpt4

### Tasks in Hindi

- Summarize
- Write like a person (known/copy style)
- Summarize
- Spatial puzzles
