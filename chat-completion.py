import os
import math
from typing import List

from openai import OpenAI
from dotenv import load_dotenv
import numpy as np
from scipy import spatial


def query_openai(input:str, ) ->List[List[int]]:
    
    # Load API key
    load_dotenv()
    openai_api_key = os.getenv('OPENAI_API')
  
    client = OpenAI(api_key=openai_api_key)

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "developer", "content": "You are a robot assistant. Please rank the following primitives to accomplish the given directive."},
            {"role": "user", "content": input}
        ],
        logprobs=True
    )
    
    print(completion)

    return completion


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
        "stop"
    ]

    prim_str = " ".join(primitives)
    input = "Directive: I'm hungry, bring me a snack. Primitives:" + prim_str

    
    output = query_openai(input)


