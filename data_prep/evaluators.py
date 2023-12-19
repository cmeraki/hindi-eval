from openai import OpenAI

from .logger import DataPrepLogger

logger = DataPrepLogger(__name__).get_logger()

class GPTEvaluator():
    def __init__(self) -> None:
        from dotenv import load_dotenv
        load_dotenv()

        self.client = OpenAI()
        self.model_id = 'gpt4'

    def evaluate(self, prompt: str, response1: str, response2: str) -> None:
        complete_msg = """The task is to {task_prompt}

        Which one of the following two reponses is better?

        1. {response1}

        2. {response2}

        Only output the number as to which one is better (1 or 2). The better
        response is
        """.format(
            prompt=prompt,
            response1=response1,
            response2=response2
        )

        completions = self.client.chat.completions.create(
            model=self.model_id,
            messages=complete_msg
        )

        logger.info(
            f'Used {completions.usage.total_tokens} tokens for {self.model_id}')

        return completions.choices[0].message.content

class Embeddor:
    def __init__(self) -> None:
        pass