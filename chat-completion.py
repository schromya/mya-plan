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
            {"role": "developer", "content": "You are an assitant for a 7-DOF Robot arm. Please rank ALL the following primitives to accomplish the given directive. Use stop to indicate the last relavent primative."},
            {"role": "user", "content": example_query},
            {"role": "assistant", "content": example_response},
            {"role": "user", "content": input}
        ],
    )


    return completion






if __name__ == "__main__":
    # primitives = [
    #     "PICK",
    #     "SCREW",
    #     "PLACE",
    #     "WIPE",
    #     "MOVE_OBJECT",
    #     "RESET",
    #     "MOVE",
    #     "MOVE_TO_CONTACT",
    #     "GRASP",
    #     "RELEASE",
    #     "VIEW",
    #     "INSPECT",
    #     "WAIT",
    #     "PUSH",
    #     "STOP",
    #     "INSERT",
    #     "UNSCREW",
    #     "MOVE_ANGLE",
    #     "PULL",
    #     "PULL_DRAWER",
    #     "DROP",
    # ]

    primitives = [
        "PICK",
        "SCREW",
        "PLACE",
        "RESET",
        "MOVE",
        "GRASP",
        "RELEASE",
        "PUSH",
        "STOP",
        "INSERT",
        "UNSCREW",
        "PULL",
        "DROP",
    ]

    prim_str = " ".join(primitives)
    input = "Directive: Insert a USB. Primitives:" + prim_str
    example_query = "Directive: Open a jar. Assume jar base is already stabalized. Primitives:" + prim_str
    example_response = "1. UNSCREW\n2. RELEASE\n3. PLACE\n4. STOP\n5. MOVE\n6. GRASP\n7. PICK\n8. RESET\n9. PUSH\n10. DROP\n11. PULL\n12. INSERT\n13. SCREW"

    
    output = query_openai(input, example_query, example_response)
    formatted_json = json.dumps(output.model_dump(), indent=4)
    # print(formatted_json)

    print(output.choices[0].message.content)


    """
    TODO:
    - Need to give description of how each primitive works
    """
