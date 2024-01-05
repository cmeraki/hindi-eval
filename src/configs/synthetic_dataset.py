from enum import Enum
from typing import Optional, Dict, Type, List, Union, Any
from pydantic import BaseModel
from dataclasses import dataclass
from textwrap import dedent

from .prompts import SystemPrompt
from ..utils.dataset_processor import (
    get_retreival_data_sys_prompt,
    get_synthetic_data_sys_prompt
)

class LANGUAGE(Enum):
    english='english'
    devnagri_hindi='devnagri_hindi'
    hinglish='hinglish'
    romanized_hindi='romanized_hindi'

class MCQResponse(BaseModel):
    QUESTION: str
    A: str
    B: str
    C: str
    D: str
    TARGET: str
    SUBJECT: Optional[str] = None
    GRADE: Optional[str] = None
    TOPIC: Optional[str] = None


class RetrievalMCQResponse(BaseModel):

    class QUESTION_TYPES(Enum):
        mcq = "mcq"
        true_false = "true_false"
        fill_in_the_blanks = "fill_in_the_blanks"

    QUESTION: str
    TYPE: Any = None
    CHOICES: Any = None
    TARGET: Any
    LANGUAGE: Any = None
    PASSAGE_LINK: Optional[str] = None

class MultiRetrievalMCQResponse(BaseModel):
    RESPONSE: List[RetrievalMCQResponse]

@dataclass
class GenerationConfiguration:
    model_id: str = 'gpt-4-1106-preview'
    temperature: float = 1.4


class OutputType(Enum):
    single = 'single'
    multi = 'multi'

class SyntheticDatasets(BaseModel):
    name: str
    # Flag to specify if the dataset should be generated
    enabled: bool
    # Number of samples to generate
    sample_size: int
    # System prompt that will be passed to GPT
    system_prompt: str
    # Will be passed as the first message after system prompt
    example_prompt: Optional[str] = None
    # Any reference that will be passed to the preprocess func
    reference_dataset: Optional[Dict] = None
    # Pydantic model that needs to be conformed to when the JSON response is returned
    response_model: Type[BaseModel]
    # JSON of response model TODO: Generate this automatically
    required_format: str 
    # Only 1 preprocess func allowed, yields system prompt and metadata dict. This should be a generator
    preprocess_func: Optional[object] = None
    # How many JSON items will be returned
    output_type: OutputType = 'single'


synthetic_dataset_subjects = {
    "Physics": {
        "5th Grade": [
            "Basic Forces and Motion",
            "Simple Machines",
            "Energy Forms and Changes",
            "Light and Shadows",
            "Sound and Its Properties",
            "Electricity and Circuits",
            "Magnetism"
        ],
        "8th Grade": [
            "Forces and Motion",
            "Energy Types and Transfer",
            "Light and Optics",
            "Sound and Waves",
            "Electricity and Magnetism",
            "Simple Machines",
            "Basics of Heat and Temperature"
        ],
        "10th Grade": [
            "Motion and Forces",
            "Work, Energy, and Power",
            "Laws of Motion",
            "Gravitation",
            "Electricity",
            "Magnetic Effects of Electric Current",
            "Optics",
        ],
        "12th Grade": [
            "Current Electricity",
            "Magnetic Effects of Current and Magnetism",
            "Electromagnetic Waves",
            "Communication Systems"
        ]
    },
    "Maths": {
        "5th Grade": [
            "Number Systems",
            "Basic Operations (Addition, Subtraction, Multiplication, Division)",
            "Fractions and Decimals",
            "Measurement (Length, Weight, Volume)",
            "Introduction to Algebra (Simple Equations)",
            "Basic Arithmetic (Addition, Subtraction, Multiplication, Division)",
            "Simple Fractions and Decimals",
            "Time and Calendar Calculations",
        ],
        "8th Grade": [
            "Rational Numbers",
            "Linear Equations in One Variable",
            "Data Handling and Probability",
            "Squares and Square Roots, Cubes and Cube Roots",
            "Comparing Quantities (Percentage, Profit and Loss)",
            "Algebraic Expressions and Identities",
            "Mensuration (Areas and Volumes)",
            "Advanced Arithmetic Operations",
            "Working with Fractions, Decimals, and Percentages",
            "Basic Algebraic Operations",
            "Ratios and Proportions",
            "Simple Interest and Compound Interest Calculations"
        ],
        "10th Grade": [
            "Real Numbers",
            "Polynomials",
            "Pair of Linear Equations in Two Variables",
            "Quadratic Equations",
            "Arithmetic Progressions",
            "Statistics and Probability",
            "Probability and Statistics Fundamentals",
            "Time, Distance, and Work Problems",
        ],
        "12th Grade": [
            "Linear Programming",
            "Probability",
        ]
    },
    "Chemistry": {
        "5th Grade": [
            "Introduction to Matter",
            "States of Matter: Solid, Liquid, Gas",
            "Basic Physical and Chemical Changes",
            "Water and Its Properties",
            "Air and Gases Around Us"
        ],
        "8th Grade": [
            "Matter and Its Nature",
            "Atomic Structure",
            "Chemical Reactions and Equations",
            "Metals and Non-metals",
            "Carbon and Its Compounds",
            "Pollution and Its Control"
        ]
    },
    "Biology": {
        "5th Grade": [
            "Living and Non-living Things",
            "Plant Life and Plant Parts",
            "Animal Life",
            "Human Body and Health",
            "Food and Nutrition",
            "Environment and Its Conservation"
        ],
        "8th Grade": [
            "Crop Production and Management",
            "Microorganisms: Friend and Foe",
            "Conservation of Plants and Animals",
            "Cell Structure and Functions",
            "Reproduction in Animals",
            "Food Production and Management"
        ]
    }
}

synthetic_dataset_models = {
    'general_mcq': SyntheticDatasets(
        name='general_mcq',
        enabled=False,
        sample_size=50,
        system_prompt=SystemPrompt.hindi_mcq,
        reference_dataset=synthetic_dataset_subjects,
        response_model=MCQResponse,
        required_format=dedent("{'QUESTION': <>, 'A': <>, 'B': <>, 'C': <>, 'D': <>, 'TARGET': <>}").strip(),
        preprocess_func=get_synthetic_data_sys_prompt
    ),
    'retrieval_questions': SyntheticDatasets(
        name='retrieval_questions',
        enabled=True,
        sample_size=100,
        system_prompt=SystemPrompt.retrieval_questions,
        response_model=MultiRetrievalMCQResponse,
        required_format=dedent("""
            'RESPONSE': [{
                'QUESTION': <str>,
                'TYPE': <str>
                'CHOICES': <List>,
                'TARGET': <int index of Choices>,
                'LANGUAGE': <str>
            }]
        """).strip(),
        preprocess_func=get_retreival_data_sys_prompt,
        output_type=OutputType.multi
    )
}