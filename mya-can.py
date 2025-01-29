import os
import math
from typing import List

from openai import OpenAI
from dotenv import load_dotenv
import numpy as np
from scipy import spatial


def query_openai(input:List[str]=[""], engine:str="text-embedding-ada-002", max_tokens:int=500, temperature:int=0.7, 
              logprobs:int=1, echo:bool=False) ->List[List[int]]:
    
    # Load API key
    load_dotenv()
    openai_api_key = os.getenv('OPENAI_API')
  
    client = OpenAI(api_key=openai_api_key)

    response = client.embeddings.create(
        input=input,
        model=engine
    )

    embeddings = []

    for data in response.data:
        embeddings.append(data.embedding)

    return embeddings


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

# From here: https://github.com/openai/openai-cookbook/blob/main/examples/utils/embeddings_utils.py
def distances_from_embeddings(
    query_embedding: List[float],
    embeddings: List[List[float]],
    distance_metric="cosine",
) -> List[List]:
    """Return the distances between a query embedding and a list of embeddings."""
    distance_metrics = {
        "cosine": spatial.distance.cosine,
        "L1": spatial.distance.cityblock,
        "L2": spatial.distance.euclidean,
        "Linf": spatial.distance.chebyshev,
    }
    distances = [
        distance_metrics[distance_metric](query_embedding, embedding)
        for embedding in embeddings
    ]
    return distances

def indices_of_nearest_neighbors_from_distances(distances) -> np.ndarray:
    """Return a list of indices of nearest neighbors from a list of distances."""
    return np.argsort(distances)
        
def rank_embeddings_v2(embeddings) -> List[int]:
    """
    TODO change this to array of dictionary
    """
    query_embedding = embeddings[0]
    embeddings = embeddings[1:]


    # get distances between the source embedding and other embeddings (function from embeddings_utils.py)
    distances = distances_from_embeddings(query_embedding, embeddings, distance_metric="cosine")

    # get indices of nearest neighbors (function from embeddings_utils.py)
    indices_of_nearest_neighbors = indices_of_nearest_neighbors_from_distances(distances)
    return indices_of_nearest_neighbors


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
    input = ["plug in a usb"] + primitives

    
    embeddings = query_openai(input)



    ranked_indices = rank_embeddings_v2(embeddings)
    print("RANKED PRIMITIVES")
    for idx in ranked_indices:
        print(primitives[idx])