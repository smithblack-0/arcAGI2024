"""
Data wrangling. Turns arc-agi examples into the zones - blocks format.
"""

# Imports
import json
from typing import Dict, List, Any

# Define files to start wrangling from

training_challenges = r"C:\Users\chris\PycharmProjects\arcAGI2024\data\raw\arc-agi_training_challenges.json"
training_solutions = r"C:\Users\chris\PycharmProjects\arcAGI2024\data\raw\arc-agi_training_solutions.json"

evaluation_challenges = r"C:\Users\chris\PycharmProjects\arcAGI2024\data\raw\arc-agi_evaluation_challenges.json"
evaluation_solutions = r"C:\Users\chris\PycharmProjects\arcAGI2024\data\raw\arc-agi_evaluation_solutions.json"

destination = r"C:\Users\chris\PycharmProjects\arcAGI2024\data\block_zone\converted_data.json"
def load_data():
    # Define outputs
    challenges = {}
    solutions = {}

    ## Gather challenges together

    with open(training_challenges, 'r') as file:
        challenges.update(json.load(file))
    with open(evaluation_challenges, 'r') as file:
        challenges.update(json.load(file))

    # Gather solutions together

    with open(training_solutions, 'r') as file:
        solutions.update(json.load(file))
    with open(evaluation_solutions, 'r') as file:
        solutions.update(json.load(file))

    # Return
    return challenges, solutions

def convert_challenge_to_scenario(challenge: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extracts the input into its train and test case.

    :param challenge: The challenge. Structured as train, test elements,
        consisting of input, output list dictionaries.
    :return:
    """
    # Define the zone blocks
    zone_blocks = []

    # Define the training cases to start with example, then show
    # the input queue and the actual input, then the output.
    try:
        challenge["train"]
    except Exception as err:
        print(challenge)

    for training_case in challenge["train"]:
        zone_blocks.append({"mode" : "text", "payload" : "example"})
        zone_blocks.append({"mode" : "text", "payload" : "input"})
        zone_blocks.append({"mode" : "intgrid", "payload" : training_case['input']})
        zone_blocks.append({"mode" : "text", "payload" : "output"})
        zone_blocks.append({"mode" : "intgrid", "payload" : training_case['output']})

    # Define the block to show the target output
    zone_blocks.append({"mode" : "text", "payload" : "unsolved case"})
    zone_blocks.append({"mode" : "intgrid", "payload" : challenge["test"][0]["input"]})

    # Return the scenario
    return zone_blocks

def convert_solution_to_scenario_answer(solution: List[List[int]])->List[Dict[str, Any]]:
    """
    Redefines the solution in terms of a collection of blocks.

    :param solution: The solution under consideration
    :return: The solution as a set of blocks
    """
    blocks = []
    blocks.append({"mode" : "text", "payload" : "Scenario Answer"})
    blocks.append({"mode" : "intgrid", "payload" : solution[0]})
    return blocks

def get_empty_case()->Dict[str, List[Dict[str, Any]]]:
    """
    Defines an object that is filled with a bunch of empty
    zones.
    :return: The setup zone.
    """
    case = {}
    case["rules_statement"] = []
    case["scenario"] = []
    case["scenario_answer"] = []
    case["rules_deduction"] = []
    case["solving_steps"] = []
    return case

def create_case(challenge: Dict[str, Any], solution: List[List[int]]) -> Dict[str, Any]:
    """Converts a challenge, solution pair into a problem case."""
    case = get_empty_case()
    case["scenario"] = convert_challenge_to_scenario(challenge)
    case["scenario_answer"] = convert_solution_to_scenario_answer(solution)
    return case

def convert_data():

    # Load the data
    challenges, solutions = load_data()

    # Setup accumulator
    cases = {}

    # Loop through and accumulate
    for key in challenges.keys():
        challenge = challenges[key]
        solution = solutions[key]
        cases[key] = create_case(challenge, solution)

    # Save it
    with open(destination, 'w') as f:
        json.dump(cases, f)

convert_data()