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

    primitives = {
        "PICK":
            "Gripper moves to the specified position and gripper closes. Once it closed, it move the object the pre-pick position.",
        "PLACE":
            "Gripper should move to the specified position and release. However, the gripper \
            won't directly move to it, but rather move to a pre-place position which is 8cm above the place position, \
            then move slowly with force control to the place position. Once it released, it will return to the pre-place \
            position. \
            Not like drop, the place action put things down slowly with care.",
        "RESET":
            "Releases the gripper and moves the arm back to the default position",
        "MOVE":
            "Moves the gripper to the specified position.",
        "GRASP":
            "Closes the gripper.",
        "RELEASE":
            "Releases the gripper.",
        "STOP":
            "Immediatly stops all movement in with the robot.",
        "UNSCREW":
            "Gripper goes to specified position, grasps the object, and untwists is 2 times, and then \
            pull the object up",
        "DROP":
            "Gripper should move to the specified position and release. However, the gripper\
            won't directly move to it, but rather move to a pre-drop position which is 8cm above the drop position,\
            then move IMMEDIATELY to the drop position. Once it released, it will return to the pre-drop \
            position.Not like place, the drop action put down things quickly."
    }

    prim_str  = " | ".join(f"{key}: {value}" for key, value in primitives.items())


    input = "Directive: Insert a USB. Primitives:" + prim_str
    example_query = "Directive: Open a jar. Assume jar base is already stabalized. Primitives:" + prim_str
    example_response = "1. MOVE\n 2. UNSCREW\n 2. PLACE\n3. STOP\n"

    
    output = query_openai(input, example_query, example_response)
    formatted_json = json.dumps(output.model_dump(), indent=4)

    print(output.choices[0].message.content)

