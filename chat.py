import os
import json
from typing import List

from openai import OpenAI
from dotenv import load_dotenv
import numpy as np
from scipy import spatial

class LLMInterface:
    def __init__(self):
        self.chat_history = [{"role": "developer", "content": "You are an assistant for a 7-DOF Robot arm. Please rank ALL the following primitives to accomplish the given directive. Use stop to indicate the last relavent primative."}]

        # Load API key
        load_dotenv()
        llm_api_key = os.getenv('OPENAI_API')
        self.client = OpenAI(api_key=llm_api_key)

    def init_history(self, example_query:str, example_response:str):
        """
        Give an example chat user query and LLM response before starting chat.
        """
        self.chat_history += [{"role": "user", "content": example_query}, {"role": "assistant", "content": example_response}]


    def query_openai(self, input:str) -> "OpenAI.ChatCompletion":
        """
        Query GPT with chat history.
        """
        self.chat_history.append({"role": "user", "content": input})
        completion = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=self.chat_history
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
            "Immediately stops all movement in with the robot.",
        "UNSCREW":
            "Gripper goes to specified position, grasps the object, and untwists is 2 times, and then \
            pull the object up",
        "DROP":
            "Gripper should move to the specified position and release. However, the gripper\
            won't directly move to it, but rather move to a pre-drop position which is 8cm above the drop position,\
            then move IMMEDIATELY to the drop position. Once it released, it will return to the pre-drop \
            position.Not like place, the drop action put down things quickly."
    }

    prim_str  = "Primitives: " + " | ".join(f"{key}: {value}" for key, value in primitives.items())

    llm = LLMInterface()
    example_query = "Directive: Open a jar. Assume jar base is already stabilized." + prim_str
    example_response = "1. MOVE\n 2. UNSCREW\n 2. PLACE\n3. STOP\n"
    llm.init_history(example_query, example_response)

    query = "Directive: Insert a USB." 
    print("\033[1m> User: \033[0m" + query)
    output = llm.query_openai(query + prim_str)
    print("\033[1m> Agent: \033[0m\n" + output.choices[0].message.content)

    while(True):
        input_query = input("\033[1m> User: \033[0m")
        output = llm.query_openai(input_query)
        print("\033[1m> Agent: \033[0m\n" + output.choices[0].message.content)

