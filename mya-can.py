import os
import math
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

    str_to_embed = []

    for embed, string in zip(embeddings, input):

        total_embed_score = 0
        embed_array = embed.embedding
        for embed_score in embed_array:
            total_embed_score += embed_score
        
        print(string, ":", total_embed_score)
        str_to_embed.append([string, total_embed_score])
    
    return str_to_embed
        

def rank_embeddings(embed_map):
    """
    TODO change this to array of dictionary
    """
    query_embed = embed_map[0][1]
    primitives = embed_map[1:]

    for prim in primitives:
        distance = math.fabs(prim[1] - query_embed)
        prim.append(distance)

    # Sort prim by distance from query embedings
    primitives = sorted(primitives, key=lambda x: x[2])

    return primitives
        



if __name__ == "__main__":
    primitives = [
        "PICK",
        "END",
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
    input = ["Open a jar"] + primitives

    
    response = query_openai(input)


    embed_map = parse_embeddings(response, input)
    print(embed_map)

    ranked_prims = rank_embeddings(embed_map)
    print("RANKED PRIMITIVES")
    for prim in ranked_prims:
        print(prim[0], prim[1], prim[2])