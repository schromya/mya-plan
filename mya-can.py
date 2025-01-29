import os

import openai
from dotenv import load_dotenv



def query_gpt(prompt:str="", engine:str="text-embedding-ada-002", max_tokens:int=500, temperature:int=0.7, 
              logprobs:int=1, echo:bool=False):
    
    # Load API key
    load_dotenv()
    openai_api_key = os.getenv('OPENAI_API')
    openai.api_key = openai_api_key

    response = openai.Completion.create(engine=engine, 
                                            prompt=prompt, 
                                        max_tokens=max_tokens, 
                                        temperature=temperature,
                                        logprobs=logprobs,
                                        echo=echo,
                                        stop=["\n\n\n"])


    return response


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

    prim_str = " ".join(primitives)
    prompt = "How would I unscrew a jar a peanut butter using the following primitives: " + prim_str

    print(prompt)
    response = query_gpt(prompt)
   

    output_file = "openai_response.txt"

    # Write the response to a text file
    with open(output_file, "w", encoding="utf-8") as file:
        file.write(str(response))

    print(f"Response saved to {output_file}")

