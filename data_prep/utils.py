import tiktoken
from typing import List, Dict

def get_tokens_from_messages(messages: List, model: str="gpt-3.5-turbo-0613") -> int:
    """
    Returns the number of tokens used by a list of messages.

    messages: List of List or List of Dict
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    assert isinstance(messages, list), f'messages should be list, {type(messages)} passed'

    if model == "gpt-3.5-turbo-0613":
        num_tokens = 0
        for message in messages:
            num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n

            if isinstance(message, dict): # Assuming a list of dicts
                for key, value in message.items():
                    num_tokens += len(encoding.encode(value))
                    if key == "name":  # if there's a name, the role is omitted
                        num_tokens += -1  # role is always required and always 1 token

            else: # Assuming a list of list
                num_tokens += len(encoding.encode(message))

        num_tokens += 2  # every reply is primed with <im_start>assistant
        return num_tokens
    else:
        raise NotImplementedError(f"""num_tokens_from_messages() is not presently implemented for model {model}.
See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.""")


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


def get_batch_tokens(example: Dict, message_key: str, model="gpt-3.5-turbo-0613") -> int:
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


def get_stringify_conversation(example: Dict, message_key: str) -> Dict:
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


def get_translate_query(example: Dict, example_english: str, example_hindi: str) -> Dict:
    system_prompt = {
        "role": "system",
        "content": "You are an expert tranlator who traslates given text in English to Devnagri Hindi"
    }

    user_example_prompt = {
        "role": "user",
        "content": f"Translate {example_english} to Devnagri Hindi"
    }

    assistant_example_prompt = {
        "role": "assistant",
        "content": f"{example_hindi}"
    }

    prompts_to_insert = [system_prompt, user_example_prompt, assistant_example_prompt]
    for p in prompts_to_insert:
        example['messages'].insert(0, prompt)

    return example