from haystack.nodes import QuestionGenerator

# Initialize the QuestionGenerator
qg = QuestionGenerator(model_name_or_path="valhalla/t5-small-qg-prepend")

# Provide input text
context = """
Wikipedia is a free online encyclopedia, created and edited by volunteers around the world and hosted by the Wikimedia Foundation.
"""

# Generate question-answer pairs
output = qg.generate(context)
print(output)
