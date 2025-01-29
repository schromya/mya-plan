import os
from typing import List

from openai import OpenAI
from dotenv import load_dotenv
import numpy as np



def query_openai(input:List[str]=[""], engine:str="text-embedding-ada-002", max_tokens:int=500, temperature:int=0.7, 
              logprobs:int=1, echo:bool=False):
    
    # Load API key
    load_dotenv()
    openai_api_key = os.getenv('OPENAI_API')
  
    client = OpenAI(api_key=openai_api_key)

    response = client.embeddings.create(
        input=input,
        model=engine
    )

    return response.data


def parse_embeddings(embeddings:List[List['OpenAI.embedding']], input:List[List[str]]):
    """
    Embedding array should be length of query + number of primitives.
    """ 

    str_to_embed = {}

    for embed, string in zip(embeddings, input):

        total_embed_score = 0
        embed_array = embed.embedding
        for embed_score in embed_array:
            total_embed_score += embed_score
        
        print(string, ":", total_embed_score)
        str_to_embed[string] = total_embed_score
    
    return str_to_embed
        





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
    input = ["How would I unscrew a jar a peanut butter?"] + primitives

    
    response = query_openai(input)


   

    output_file = "openai_response.txt"

    # Write the response to a text file
    with open(output_file, "w", encoding="utf-8") as file:
        file.write(str(response))

    print(f"Response saved to {output_file}")


    embed_map = parse_embeddings(response, input)
    print(embed_map)
