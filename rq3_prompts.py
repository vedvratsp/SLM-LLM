"""
rq3_prompts.py — Six custom prompts used in the RQ3 human perception survey.

These prompts were designed to sample across MT-Bench evaluation categories
(Zheng et al., 2023) and form the basis of the ethics-approved human study.
They are NOT the actual MT-Bench items; see mt_bench_prompts.py for those.

Ethics reference: TU Dublin REIC approval, protocol as submitted.
"""

RQ3_PROMPTS = [
    {
        "id": 1,
        "category": "reasoning",
        "text": (
            "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. "
            "How much does the ball cost? Show your reasoning step by step."
        ),
    },
    {
        "id": 2,
        "category": "stem",
        "text": (
            "Explain the difference between machine learning and deep learning. "
            "When would you choose one over the other?"
        ),
    },
    {
        "id": 3,
        "category": "writing",
        "text": (
            "Write a professional email to a client informing them that their project deadline "
            "will be delayed by two weeks due to unexpected technical issues."
        ),
    },
    {
        "id": 4,
        "category": "advice",
        "text": (
            "I work from home and find it hard to stay focused. "
            "What are three practical strategies to improve my productivity?"
        ),
    },
    {
        "id": 5,
        "category": "summarisation",
        "text": (
            "Summarise the main causes and effects of climate change in under 200 words, "
            "suitable for a general audience."
        ),
    },
    {
        "id": 6,
        "category": "ethics",
        "text": (
            "Is it ever morally acceptable to lie to protect someone's feelings? "
            "Discuss both sides and give your view."
        ),
    },
]

# Convenience alias
PROMPTS = RQ3_PROMPTS
