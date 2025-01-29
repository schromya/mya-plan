





import collections
import datetime
import os
import random
import threading
import time

from dotenv import load_dotenv
import cv2  # Used by ViLD.
import imageio
from heapq import nlargest

# import matplotlib.pyplot as plt
import numpy as np
import openai

import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt

ENGINE = "text-embedding-ada-002" # "gpt-4o-mini"  

# Load API key
load_dotenv()
openai_api_key = os.getenv('OPENAI_API')


openai.api_key = openai_api_key



#@markdown Global constants: pick and place objects, colors, workspace bounds

PICK_TARGETS = {
  "blue block": None,
  "red block": None,
  "green block": None,
  "yellow block": None,
}

COLORS = {
    "blue":   (78/255,  121/255, 167/255, 255/255),
    "red":    (255/255,  87/255,  89/255, 255/255),
    "green":  (89/255,  169/255,  79/255, 255/255),
    "yellow": (237/255, 201/255,  72/255, 255/255),
}

PLACE_TARGETS = {
  "blue block": None,
  "red block": None,
  "green block": None,
  "yellow block": None,

  "blue bowl": None,
  "red bowl": None,
  "green bowl": None,
  "yellow bowl": None,

  "top left corner":     (-0.3 + 0.05, -0.2 - 0.05, 0),
  "top right corner":    (0.3 - 0.05,  -0.2 - 0.05, 0),
  "middle":              (0,           -0.5,        0),
  "bottom left corner":  (-0.3 + 0.05, -0.8 + 0.05, 0),
  "bottom right corner": (0.3 - 0.05,  -0.8 + 0.05, 0),
}

PIXEL_SIZE = 0.00267857
BOUNDS = np.float32([[-0.3, 0.3], [-0.8, -0.2], [0, 0.15]])  # X Y Z



overwrite_cache = True
if overwrite_cache:
  LLM_CACHE = {}


#@title LLM Scoring

def gpt3_call(engine="text-ada-001", prompt="", max_tokens=128, temperature=0, 
              logprobs=1, echo=False):
  full_query = ""
  for p in prompt:
    full_query += p
  id = tuple((engine, full_query, max_tokens, temperature, logprobs, echo))
  if id in LLM_CACHE.keys():
    print('cache hit, returning')
    response = LLM_CACHE[id]
  else:
    response = openai.Completion.create(engine=engine, 
                                        prompt=prompt, 
                                        max_tokens=max_tokens, 
                                        temperature=temperature,
                                        logprobs=logprobs,
                                        echo=echo)

    print(response)

    LLM_CACHE[id] = response
  return response

def gpt3_scoring(query, options, engine="text-ada-001", limit_num_options=None, option_start="\n", verbose=False, print_tokens=False):
  if limit_num_options:
    options = options[:limit_num_options]
  verbose and print("Scoring", len(options), "options")
  gpt3_prompt_options = [query + option for option in options]
  response = gpt3_call(
      engine=engine, 
      prompt=gpt3_prompt_options, 
      max_tokens=0,
      logprobs=1, 
      temperature=0,
      echo=True,)
  
  scores = {}
  for option, choice in zip(options, response["choices"]):
    tokens = choice["logprobs"]["tokens"]
    token_logprobs = choice["logprobs"]["token_logprobs"]

    total_logprob = 0
    for token, token_logprob in zip(reversed(tokens), reversed(token_logprobs)):
      print_tokens and print(token, token_logprob)
      if option_start is None and not token in option:
        break
      if token == option_start:
        break

      if token_logprob:
        total_logprob += token_logprob
    scores[option] = total_logprob

  for i, option in enumerate(sorted(scores.items(), key=lambda x : -x[1])):
    verbose and print(option[1], "\t", option[0])
    if i >= 10:
      break

  return scores, response

def make_options(pick_targets=None, place_targets=None, options_in_api_form=True, termination_string="done()"):
  if not pick_targets:
    pick_targets = PICK_TARGETS
  if not place_targets:
    place_targets = PLACE_TARGETS
  options = []
  for pick in pick_targets:
    for place in place_targets:
      if options_in_api_form:
        option = "robot.pick_and_place({}, {})".format(pick, place)
      else:
        option = "Pick the {} and place it on the {}.".format(pick, place)
      options.append(option)

  options.append(termination_string)
  print("Considering", len(options), "options")
  return options


def build_scene_description(found_objects, block_name="box", bowl_name="circle"):
  scene_description = f"objects = {found_objects}"
  scene_description = scene_description.replace(block_name, "block")
  scene_description = scene_description.replace(bowl_name, "bowl")
  scene_description = scene_description.replace("'", "")
  return scene_description

def step_to_nlp(step):
  step = step.replace("robot.pick_and_place(", "")
  step = step.replace(")", "")
  pick, place = step.split(", ")
  return "Pick the " + pick + " and place it on the " + place + "."

def normalize_scores(scores):
  max_score = max(scores.values())  
  normed_scores = {key: np.clip(scores[key] / max_score, 0, 1) for key in scores}
  return normed_scores

def plot_saycan(llm_scores, vfs, combined_scores, task, correct=True, show_top=None):
  if show_top:
    top_options = nlargest(show_top, combined_scores, key = combined_scores.get)
    # add a few top llm options in if not already shown
    top_llm_options = nlargest(show_top // 2, llm_scores, key = llm_scores.get)
    for llm_option in top_llm_options:
      if not llm_option in top_options:
        top_options.append(llm_option)
    llm_scores = {option: llm_scores[option] for option in top_options}
    vfs = {option: vfs[option] for option in top_options}
    combined_scores = {option: combined_scores[option] for option in top_options}

  sorted_keys = dict(sorted(combined_scores.items()))
  keys = [key for key in sorted_keys]
  positions = np.arange(len(combined_scores.items()))
  width = 0.3

  fig = plt.figure(figsize=(12, 6))
  ax1 = fig.add_subplot(1,1,1)

  plot_llm_scores = normalize_scores({key: np.exp(llm_scores[key]) for key in sorted_keys})
  plot_llm_scores = np.asarray([plot_llm_scores[key] for key in sorted_keys])
  plot_affordance_scores = np.asarray([vfs[key] for key in sorted_keys])
  plot_combined_scores = np.asarray([combined_scores[key] for key in sorted_keys])
  
  ax1.bar(positions, plot_combined_scores, 3 * width, alpha=0.6, color="#93CE8E", label="combined")
    
  score_colors = ["#ea9999ff" for score in plot_affordance_scores]
  ax1.bar(positions + width / 2, 0 * plot_combined_scores, width, color="#ea9999ff", label="vfs")
  ax1.bar(positions + width / 2, 0 * plot_combined_scores, width, color="#a4c2f4ff", label="language")
  ax1.bar(positions - width / 2, np.abs(plot_affordance_scores), width, color=score_colors)
  
  plt.xticks(rotation="vertical")
  ax1.set_ylim(0.0, 1.0)

  ax1.grid(True, which="both")
  ax1.axis("on")

  ax1_llm = ax1.twinx()
  ax1_llm.bar(positions + width / 2, plot_llm_scores, width, color="#a4c2f4ff", label="language")
  ax1_llm.set_ylim(0.01, 1.0)
  plt.yscale("log")
  
  font = {"fontname":"Arial", "size":"16", "color":"k" if correct else "r"}
  plt.title(task, **font)
  key_strings = [key.replace("robot.pick_and_place", "").replace(", ", " to ").replace("(", "").replace(")","") for key in keys]
  plt.xticks(positions, key_strings, **font)
  ax1.legend()
  plt.show()


#@title Affordance Scoring
#@markdown Given this environment does not have RL-trained policies or an asscociated value function, we use affordances through an object detector.

def affordance_scoring(options, found_objects, verbose=False, block_name="box", bowl_name="circle", termination_string="done()"):
  affordance_scores = {}
  found_objects = [
                   found_object.replace(block_name, "block").replace(bowl_name, "bowl") 
                   for found_object in found_objects + list(PLACE_TARGETS.keys())[-5:]]
  verbose and print("found_objects", found_objects)
  for option in options:
    if option == termination_string:
      affordance_scores[option] = 0.2
      continue
    pick, place = option.replace("robot.pick_and_place(", "").replace(")", "").split(", ")
    affordance = 0
    found_objects_copy = found_objects.copy()
    if pick in found_objects_copy:
      found_objects_copy.remove(pick)
      if place in found_objects_copy:
        affordance = 1
    affordance_scores[option] = affordance
    verbose and print(affordance, '\t', option)
  return affordance_scores


if __name__ == "__main__":
  #@title Prompt

  termination_string = "done()"

  gpt3_context = """
  objects = [red block, yellow block, blue block, green bowl]
  # move all the blocks to the top left corner.
  robot.pick_and_place(blue block, top left corner)
  robot.pick_and_place(red block, top left corner)
  robot.pick_and_place(yellow block, top left corner)
  done()

  objects = [red block, yellow block, blue block, green bowl]
  # put the yellow one the green thing.
  robot.pick_and_place(yellow block, green bowl)
  done()

  objects = [yellow block, blue block, red block]
  # move the light colored block to the middle.
  robot.pick_and_place(yellow block, middle)
  done()

  objects = [blue block, green bowl, red block, yellow bowl, green block]
  # stack the blocks.
  robot.pick_and_place(green block, blue block)
  robot.pick_and_place(red block, green block)
  done()

  objects = [red block, blue block, green bowl, blue bowl, yellow block, green block]
  # group the blue objects together.
  robot.pick_and_place(blue block, blue bowl)
  done()

  objects = [green bowl, red block, green block, red bowl, yellow bowl, yellow block]
  # sort all the blocks into their matching color bowls.
  robot.pick_and_place(green block, green bowl)
  robot.pick_and_place(red block, red bowl)
  robot.pick_and_place(yellow block, yellow bowl)
  done()
  """

  use_environment_description = False
  gpt3_context_lines = gpt3_context.split("\n")
  gpt3_context_lines_keep = []
  for gpt3_context_line in gpt3_context_lines:
    if "objects =" in gpt3_context_line and not use_environment_description:
      continue
    gpt3_context_lines_keep.append(gpt3_context_line)

  gpt3_context = "\n".join(gpt3_context_lines_keep)
  print(gpt3_context)



  raw_input = "put all the blocks in different corners." 
  config = {"pick":  ["red block", "yellow block", "green block", "blue block"],
            "place": ["red bowl"]}
  


  plot_on = True
  max_tasks = 5

  options = make_options(PICK_TARGETS, PLACE_TARGETS, termination_string=termination_string)
  # Temporarily have all objects
  found_objects = ['blue block',
                    'red block',
                    'green block',
                    # 'orange block',
                    # 'yellow block',
                    # 'purple block',
                    # 'pink block',
                    # 'cyan block',
                    # 'brown block',
                    'gray block',

                    'blue bowl',
                    'red bowl',
                    # 'green bowl',
                    # 'orange bowl',
                    # 'yellow bowl',
                    # 'purple bowl',
                    # 'pink bowl',
                    # 'cyan bowl',
                    'brown bowl',
                    'gray bowl']
  
  scene_description = build_scene_description(found_objects)
  env_description = scene_description

  print(scene_description)

  gpt3_prompt = gpt3_context
  if use_environment_description:
    gpt3_prompt += "\n" + env_description
  gpt3_prompt += "\n# " + raw_input + "\n"

  all_llm_scores = []
  all_affordance_scores = []
  all_combined_scores = []
  affordance_scores = affordance_scoring(options, found_objects, block_name="box", bowl_name="circle", verbose=False)
  num_tasks = 0
  selected_task = ""
  steps_text = []
  while not selected_task == termination_string:
    num_tasks += 1
    if num_tasks > max_tasks:
      break

    llm_scores, _ = gpt3_scoring(gpt3_prompt, options, verbose=True, engine=ENGINE, print_tokens=False)
    combined_scores = {option: np.exp(llm_scores[option]) * affordance_scores[option] for option in options}
    combined_scores = normalize_scores(combined_scores)
    selected_task = max(combined_scores, key=combined_scores.get)
    steps_text.append(selected_task)
    print(num_tasks, "Selecting: ", selected_task)
    gpt3_prompt += selected_task + "\n"

    all_llm_scores.append(llm_scores)
    all_affordance_scores.append(affordance_scores)
    all_combined_scores.append(combined_scores)

  if plot_on:
    for llm_scores, affordance_scores, combined_scores, step in zip(
        all_llm_scores, all_affordance_scores, all_combined_scores, steps_text):
      plot_saycan(llm_scores, affordance_scores, combined_scores, step, show_top=10)

  print('**** Solution ****')
  print(env_description)
  print('# ' + raw_input)
  for i, step in enumerate(steps_text):
    if step == '' or step == termination_string:
      break
    print('Step ' + str(i) + ': ' + step)
    nlp_step = step_to_nlp(step)

