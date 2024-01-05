from textwrap import dedent
import openai
import backoff
from openai import OpenAI

from .utils.logger import DataPrepLogger

logger = DataPrepLogger(__name__).get_logger()

class GPTEvaluator():
    def __init__(self, model_id) -> None:
        from dotenv import load_dotenv
        load_dotenv()

        self.client = OpenAI()
        self.model_id = model_id

    @backoff.on_exception(backoff.expo, openai.RateLimitError, max_time=300)
    def evaluate(self, original_text:str, candidate_1: str, candidate_2: str) -> None:
        sys_prompt = {
            'role': 'system',
            'content': dedent('''
                You are an expert judge/evaluator. You will evaluate the translation of English text to colloquial Devnagri Hindi text.
                You will be given original English text between """<english text>""" and two candidate translations of the text in Devnagri Hindi.
                Of the two candidate translations, select the better one and only output a number indicating which one is better (1 or 2).
                Return 3 if both the of the translators are equally good.
            ''').strip()
        }

        usr_prompt = {
            'role': 'user',
            'content': dedent(f'''
            """{original_text}"""

            1. {candidate_1}
            2. {candidate_2}
            ''').strip()
        }

        complete_msg = [sys_prompt, usr_prompt]

        logger.debug(f'Evaluation prompt: {complete_msg}')

        completions = self.client.chat.completions.create(
            model=self.model_id,
            messages=complete_msg
        )

        logger.debug(f'Used {completions.usage.total_tokens} tokens for {self.model_id}')

        return completions.choices[0].message.content

class Embeddor:
    def __init__(self) -> None:
        pass