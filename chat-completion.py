import os
import json
from typing import List

from openai import OpenAI
from dotenv import load_dotenv
import numpy as np
from scipy import spatial


def query_openai(input:str, example_query:str, example_response:str) -> "OpenAI.ChatCompletition":
    
    # Load API key
    load_dotenv()
    openai_api_key = os.getenv('OPENAI_API')
  
    client = OpenAI(api_key=openai_api_key)

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "developer", "content": "You are a robot assistant. Please rank the following primitives to accomplish the given directive."},
            {"role": "user", "content": example_query},
            {"role": "assistant", "content": example_response},
            {"role": "user", "content": input}
        ],
        logprobs=True
    )


    return completion

def score_openai_query(openai_response):
    """
    1. Combine tokens back into steps based off of newline delimeter. 
       Sum tokens logprogs for each step.
    2. Re-rank based off logprog (highest->lowest)
    """

    logprobs = openai_response.choices[0].logprobs.content

    steps = []  # Array of tuples (step_string, total_logprog)
    total_prob = 0
    curr_step = ""
    for i, prob in enumerate(logprobs):
        if "\n" in prob.token or i == len(logprobs) - 1:
            steps.append((curr_step, total_prob))
            total_prob = 0
            curr_step = ""
        #Skep "1.", "2.", etc
        elif prob.token.isdigit() or "." in prob.token:
            continue
        else:
            curr_step += prob.token
            total_prob += prob.logprob
    
    # Now sort steps
    sorted_steps  = sorted(steps, key=lambda x: x[1])
    print(steps)
    print(sorted_steps)
    return sorted_steps
    
        




if __name__ == "__main__":
    primitives = [
        "PICK",
        "SCREW",
        "PLACE",
        "WIPE",
        "MOVE_OBJECT",
        "RESET",
        "MOVE",
        "MOVE_TO_CONTACT",
        "GRASP",
        "RELEASE",
        "VIEW",
        "INSPECT",
        "WAIT",
        "PUSH",
        "STOP",
        "INSERT",
        "UNSCREW",
        "MOVE_ANGLE",
        "PULL",
        "PULL_DRAWER",
        "DROP",
    ]


    primitives = [
        "Pick up apple",
        "Bring me apple",
        "Pick up water",
        "Bring me water",
        "Pick up screwdriver",
        "Bring me screwdriver",
        "Stop"
    ]

    prim_str = " ".join(primitives)
    input = "Directive: I'm hungry, bring me a snack. Primitives:" + prim_str
    example_query = "Directive: I need to loosen this bolt, bring me a tool. Primitives:" + prim_str
    example_response = "1. Pick up screwdriver \n 2. Bring me screwdriver \n 3. Stop \n 4. Pick up apple \n 5. Bring me apple \n 6.Pick up water \n 7.Bring me water \n"

    
    output = query_openai(input, example_query, example_response)
    formatted_json = json.dumps(output.model_dump(), indent=4)
    print(formatted_json)
    ranked = score_openai_query(output)


