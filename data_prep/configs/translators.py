from typing import Dict
from pydantic import BaseModel, ConfigDict

from ..translators import (
    SeamlessM4TTranslator,
    GPTTranslator,
    GeminiTranslator,
    HFTranslator,
    BaseTranslator
)

class Translators(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    engine: BaseTranslator
    enabled: bool
    kwargs: Dict = {} # Passed to the model loader

translator_engines = {
    'gpt': Translators(
        engine=GPTTranslator('gpt-3.5-turbo-1106'),
        enabled=False
    ),
    'gemini': Translators(
        engine=GeminiTranslator('gemini-pro'),
        enabled=False
    ),
    'seamless_m4t': Translators(
        engine=SeamlessM4TTranslator('facebook/hf-seamless-m4t-large'),
        enabled=False
    ),
    'llama_7b_gptq': Translators(
        engine=HFTranslator('TheBloke/Llama-2-7B-Chat-GPTQ'),
        enabled=True,
        kwargs={'use_flash_attention_2': True}
    )
}
