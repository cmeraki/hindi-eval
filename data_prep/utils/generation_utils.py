import time
import tiktoken
from typing import List
from datasets import DatasetDict

from .logger import DataPrepLogger

logger = DataPrepLogger(__name__).get_logger()

GPT_MODELS = [
    'gpt-3.5-turbo-0613',
    'gpt-3.5-turbo-16k-0613',
    'gpt-4-0314',
    'gpt-4-32k-0314',
    'gpt-4-0613',
    'gpt-4-32k-0613',
]

def num_tokens_from_messages(messages: List, model: str="gpt-3.5-turbo-0613") -> int:
    """Return the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model in GPT_MODELS:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif "gpt-3.5-turbo" in model:
        print("Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613.")
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
    elif "gpt-4" in model:
        print("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
        return num_tokens_from_messages(messages, model="gpt-4-0613")
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
        )

    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name

    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens


def stringify(json_msg: str) -> str:
    """
    Converts jsonified message into flat text message

    From
    {"role": "assistant", "content": "my name is alexa"}
    To
    assistant

    my name is alexa
    """

    complete_msg = json_msg['role']
    complete_msg += '\n'
    complete_msg += json_msg['content']
    complete_msg += '\n\n'

    return complete_msg


def get_batch_tokens(example: DatasetDict, message_key: str, model="gpt-3.5-turbo-0613") -> int:
    """
    Returns the number of tokens used by the messages

    To be used with HF datasets
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    num_tokens = 0
    for message in example[message_key]:
        num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
    
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":  # if there's a name, the role is omitted
                num_tokens += -1  # role is always required and always 1 token
    
        num_tokens += 2  # every reply is primed with <im_start>assistant
    example['num_tokens'] = num_tokens
    return example


def get_stringify_conversation(example: DatasetDict, message_key: str) -> DatasetDict:
    """
    Converts messages into a single stringified conversation
    Adds to a HF dataset, to be used with HF dataset

    Args:
        example (Dict): HF dataset
        message_key (str): Message key to use

    Returns:
        Dict
    """
    str_msg = ""
    for message in example[message_key]:
        str_msg += stringify(message)

    example['str_message'] = str_msg
    return example


def get_num_convo(example: DatasetDict) -> DatasetDict:
    """
    Returns the number of chat conversations in multi turn conversations

    Args:
        example (Dict): Huggingface datasets

    Returns:
        Huggingface datasets with `num_convo` key
    """
    example['num_convo'] = len(example['messages'])
    return example


def get_prompt(example: DatasetDict, message_key: str) -> DatasetDict:
    """
    Converts multi turn conversations to single turn conversations that
    needs to be translated

    Args:
        None

    Returns:
        List: List of messages
    """
    system_prompt = {
        "role": "system",
        "content": "You are an expert tranlator who traslates given text in English to colloquial Devnagri Hindi. You output nothing except the translation."
    }

    example_prompts = [
        {
            "role": "user",
            "content": f"Which famous landmarks should I visit in London, beyond the usual ones?"
        },
        {
            "role": "assistant",
            "content": f"लंदन में मुझे कौन से प्रसिद्ध स्थल देखने चाहिए, जो आमतौर पर नहीं होते हैं?"
        },
        {
            "role": "user",
            "content": "Here is an offbeat and lesser-known place in London that locals might recommend: God's Own Junkyard - a neon wonderland filled with vintage and new neon signs. There are many other hidden gems in London, and a quick Google search for ‘offbeat things in London’ will bring up many blogs and resources with more options."
        },
        {
            "role": "assistant",
            "content": " यहां लंदन में एक अनोखा और कम जाना-पहचाना स्थल है जिसकी स्थानीय लोग सिफारिश कर सकते हैं: गॉड्स ओन जंकयार्ड - एक नियॉन वंडरलैंड जो पुराने और नए नियॉन साइन्स से भरा हुआ है। लंदन में कई अन्य छुपे हुए रत्न हैं, और ‘लंदन में अनोखी चीजें’ के लिए गूगल सर्च करने पर आपको कई ब्लॉग्स और संसाधन मिल जाएंगे जिनमें और भी विकल्प होंगे।"
        }
    ]

    complete_prompt = [
        system_prompt
    ] + example_prompts

    if isinstance(example[message_key], list):
        for m in example[message_key]:
            complete_prompt += [{
                'role': 'user',
                'content': m['content'].replace('\n', ' ')
            }]

            yield complete_prompt
            _ = complete_prompt.pop()
    else:
        pass # implement in future
        yield complete_prompt


def get_translate_query(example: DatasetDict, message_key: str) -> DatasetDict:

    for message in example[message_key]:
        yield message['content']


def time_logger(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.debug(f"Execution time of {func.__name__}: {end_time - start_time} seconds")
        return result
    return wrapper
