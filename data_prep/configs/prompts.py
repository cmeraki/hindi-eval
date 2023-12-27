from dataclasses import dataclass
from textwrap import dedent

@dataclass
class SystemPrompt:
    hindi_mcq: str = dedent("""
        You are an teacher's assistant who produces question paper in {language}.
        You generate a completely new MCQ question for {grade} {subject} on the topic of {topic}.
        You only reply with a new question, 4 answer choices and the correct answer.
        You always output in JSON format. following the format: {required_format}
    """).strip()
