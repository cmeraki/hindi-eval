from typing import Optional, Dict, Type
from pydantic import BaseModel
from dataclasses import dataclass
from textwrap import dedent

from .prompts import SystemPrompt


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


@dataclass
class GenerationConfiguration:
    model_id: str = 'gpt-3.5-turbo-1106'
    temperature: float = 1.4


class SyntheticDatasets(BaseModel):
    name: str
    system_prompt: str
    sample_size: int
    example_prompt: Optional[str] = None # Will be passed as the first message after system prompt
    reference_dataset: Optional[Dict] = None
    response_model: Type[BaseModel]
    required_format: str

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
            "Thermodynamics",
            "Electricity",
            "Magnetic Effects of Electric Current",
            "Optics",
            "Atomic and Nuclear Physics"
        ],
        "12th Grade": [
            "Electrostatics",
            "Current Electricity",
            "Magnetic Effects of Current and Magnetism",
            "Electromagnetic Induction and Alternating Currents",
            "Electromagnetic Waves",
            "Optics",
            "Dual Nature of Radiation and Matter",
            "Atoms and Nuclei",
            "Electronic Devices",
            "Communication Systems"
        ]
    },
    "Maths": {
        "5th Grade": [
            "Number Systems",
            "Basic Operations (Addition, Subtraction, Multiplication, Division)",
            "Fractions and Decimals",
            "Measurement (Length, Weight, Volume)",
            "Geometry (Basic Shapes, Angles)",
            "Introduction to Algebra (Simple Equations)",
            "Data Handling and Graphs"
        ],
        "8th Grade": [
            "Rational Numbers",
            "Linear Equations in One Variable",
            "Understanding Quadrilaterals and Practical Geometry",
            "Data Handling and Probability",
            "Squares and Square Roots, Cubes and Cube Roots",
            "Comparing Quantities (Percentage, Profit and Loss)",
            "Algebraic Expressions and Identities",
            "Mensuration (Areas and Volumes)"
        ],
        "10th Grade": [
            "Real Numbers",
            "Polynomials",
            "Pair of Linear Equations in Two Variables",
            "Quadratic Equations",
            "Arithmetic Progressions",
            "Triangles",
            "Coordinate Geometry",
            "Introduction to Trigonometry and its Applications",
            "Circles",
            "Constructions",
            "Areas Related to Circles",
            "Surface Areas and Volumes",
            "Statistics and Probability"
        ],
        "12th Grade": [
            "Relations and Functions",
            "Algebra (Matrices, Determinants)",
            "Calculus (Limits, Derivatives, Integrals, Differential Equations)",
            "Vectors and Three-Dimensional Geometry",
            "Linear Programming",
            "Probability",
            "Continuity and Differentiability",
            "Application of Derivatives",
            "Integrals",
            "Application of the Integrals",
            "Differential Equations",
            "Vector Algebra",
            "Three-dimensional Geometry"
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
        ],
        "10th Grade": [
            "Chemical Reactions and Equations",
            "Acids, Bases, and Salts",
            "Metals and Non-metals",
            "Carbon and Its Compounds",
            "Periodic Classification of Elements",
            "Basic Principles of Organic Chemistry",
            "Environmental Chemistry"
        ],
        "12th Grade": [
            "Solid State",
            "Solutions",
            "Electrochemistry",
            "Chemical Kinetics",
            "Surface Chemistry",
            "General Principles and Processes of Isolation of Elements",
            "p-Block Elements",
            "d and f Block Elements",
            "Coordination Compounds",
            "Haloalkanes and Haloarenes",
            "Alcohols, Phenols, and Ethers",
            "Aldehydes, Ketones, and Carboxylic Acids",
            "Organic Compounds Containing Nitrogen",
            "Biomolecules",
            "Polymers",
            "Chemistry in Everyday Life"
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
            "Reaching the Age of Adolescence",
            "Food Production and Management"
        ],
        "10th Grade": [
            "Life Processes",
            "Control and Coordination in Animals and Plants",
            "How do Organisms Reproduce?",
            "Heredity and Evolution",
            "Our Environment",
            "Management of Natural Resources"
        ],
        "12th Grade": [
            "Reproduction in Organisms",
            "Sexual Reproduction in Flowering Plants",
            "Human Reproduction",
            "Reproductive Health",
            "Principles of Inheritance and Variation",
            "Molecular Basis of Inheritance",
            "Evolution",
            "Human Health and Disease",
            "Strategies for Enhancement in Food Production",
            "Microbes in Human Welfare",
            "Biotechnology: Principles and Processes",
            "Biotechnology and Its Applications",
            "Organisms and Populations",
            "Ecosystem",
            "Biodiversity and Conservation",
            "Environmental Issues"
        ]
    },
    "Mental Maths": {
        "5th Grade": [
            "Basic Arithmetic (Addition, Subtraction, Multiplication, Division)",
            "Simple Fractions and Decimals",
            "Estimation Techniques",
            "Basic Geometric Shapes and Their Properties",
            "Time and Calendar Calculations",
            "Mental Calculation Strategies"
        ],
        "8th Grade": [
            "Advanced Arithmetic Operations",
            "Working with Fractions, Decimals, and Percentages",
            "Mental Estimation and Approximation",
            "Basic Algebraic Operations",
            "Ratios and Proportions",
            "Simple Interest and Compound Interest Calculations"
        ],
        "10th Grade": [
            "Speed Mathematics Techniques",
            "Advanced Algebraic Expressions and Equations",
            "Geometry and Trigonometry Basics",
            "Probability and Statistics Fundamentals",
            "Time, Distance, and Work Problems",
            "Mental Calculations in Practical Life Scenarios"
        ],
        "12th Grade": [
            "Advanced Arithmetic and Algebra Techniques",
            "Calculus Basics (Limits, Derivatives)",
            "Advanced Geometry and Trigonometry",
            "Mathematical Reasoning and Logic",
            "Complex Numbers and Quadratic Equations",
            "Data Interpretation and Analysis Techniques"
        ]
    }
}

synthetic_dataset_models = {
    'general_mcq': SyntheticDatasets(
        name='general_mcq',
        sample_size=10,
        system_prompt=SystemPrompt.hindi_mcq,
        reference_dataset=synthetic_dataset_subjects,
        response_model=MCQResponse,
        required_format=dedent("{'QUESTION': <>, 'A': <>, 'B': <>, 'C': <>, 'D': <>, 'TARGET': <>}").strip()
    )
}