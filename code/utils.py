import os
from os.path import join
import re
import ast
import torch
import json
import openai
# if openai.__verion__ >= 1.0.0:
from openai import OpenAI
import asyncio
import string
from string import ascii_uppercase as alc
# from openai import OpenAI, AsyncOpenAI
from config import openai_api_key, openai_api_base, question_placeholder, \
                   choice_placeholder, icl_cot_dir
import pdb

openai.api_key = openai_api_key
openai.api_base = openai_api_base
os.environ["OPENAI_API_KEY"] = openai_api_key

def formulate_query(query, template, attach_ans=False):
    # formulating the textual query from structured data
    # attach_ans is set to True is answer is attached to the query, otherwise only the question
    # the template includes [QUESTION] which would be replaced by question and [ANSWER] which would be replaced by answer
    return template.replace("[QUESTION]", query['question']).replace("[ANSWER]", query['answer']) if attach_ans else template.replace("[QUESTION]", query['question'])

def json_reader(file_name):
    # load the json file while one json takes more than one line of text
    with open(file_name, "r") as fp:
        result_lines = [x.strip("\n") for x in fp.readlines() if len(x.strip("\n"))]

        result = []

        json_buffer_string = ""
        for line in result_lines:
            if line == "{":
                continue
            elif line == "}":
                json_buffer = "{" + json_buffer_string.replace("null", "None").replace("true", "True").replace("false", "False") + "}"
                result.append(ast.literal_eval(json_buffer))
                json_buffer_string = ""
            else:
                json_buffer_string += line.lstrip()

    return result

def shorten_expr(expr):
    # shorten the expr with multiple "="
    segs = expr.split("=")
    return "=".join([segs[0], segs[-1]])

def get_nested_expr(queries):
    all_exprs_nested = []

    for x in queries:
        all_exprs_nested += x

    all_eqs_nested = [x for x in all_exprs_nested if "=" in x]
    all_eqs_nested_shorten = [shorten_expr(x) if x.count("=") > 1 else x for x in all_eqs_nested]    
    return all_eqs_nested_shorten

def fill_mrc_answers(candidates, symbols):
    # combine the question and the multiple choice options
    result =  ""
    for choice, candidate in zip(symbols, candidates):
        result += " {}. {}".format(choice, candidate)
    return result

def get_text_from_openai_response(response, return_usage=False, use_vllm=False):
    if use_vllm:
        # vllm
        content = response.outputs[0].text
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    elif 'choices' in response:
        content = response['choices'][0]['message']['content']
        # a json with keys "prompt_token", "completion_tokens", "total_tokens"
        usage = response['usage']
    else:
        # fastchat server
        content = response.choices[0].message.content
        usage = response.usage
        usage = {"prompt_tokens": usage.prompt_tokens, "completion_tokens": usage.completion_tokens, "total_tokens": usage.total_tokens}
    if not return_usage:
        return content
    return content, usage

async def dispatch_openai_requests_advanced(
    messages_list,
    client,
    model,
    temperature,
    max_tokens,
    top_p,
    stop_tokens,
    limiter,
):
    """Dispatches requests to OpenAI API asynchronously. For openai >= 1.0.0
    
    Args:
        messages_list: List of messages to be sent to OpenAI ChatCompletion API.
        model: OpenAI model to use.
        temperature: Temperature to use for the model.
        max_tokens: Maximum number of tokens to generate.
        top_p: Top p to use for the model.
    Returns:
        List of responses from OpenAI API.
    """

    async_responses = [
        client.chat.completions.create(
            model=model,
            messages=x,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop_tokens,
            top_p=top_p,
        )
        for x in messages_list
    ]

    try:
        return await asyncio.gather(*async_responses)
    except openai.RateLimitError as e:
        retry_time = e.retry_after if hasattr(e, 'retry_after') else 30
        print(f"Rate limit exceeded. Retrying in {retry_time} seconds...")
        await asyncio.sleep(retry_time)
        return dispatch_openai_requests_advanced(
                messages_list, client, model, temperature, 
                max_tokens, top_p, stop_tokens, limiter)
    except openai.APIError as e:
        # fix: https://github.com/langchain-ai/langchain/issues/13368
        retry_time = e.retry_after if hasattr(e, 'retry_after') else 30
        print(f"API error occurred. Retrying in {retry_time} seconds...")
        await asyncio.sleep(retry_time)
        return dispatch_openai_requests_advanced(
                messages_list, client, model, temperature, 
                max_tokens, top_p, stop_tokens, limiter)
    except openai.APIConnectionError as e:
        print("The server could not be reached")
        print(e.__cause__)  # an underlying Exception, likely raised within httpx.
    except openai.APIStatusError as e:
        print("Another non-200-range status code was received")
        print("status code: ", e.status_code)
        print(e.response)

    return async_responses

def get_icl_data(task, n_shots):
    # collect in-context learning examples for specific task and dataset
    data_file = join(icl_cot_dir, "task={}.jsonl".format(task))

    if not os.path.exists(data_file):
        print("icl data not found in {} ...".format(data_file))

    example_data = []
    with open(data_file, "r") as fp:
        example_data = [json.loads(x) for x in fp.readlines()]
    return example_data[:n_shots]

def construct_example(question, answer, hint, template, task_name, add_answer=False):
    if task_name == "quant_reason_mrc":
        question, candidates = question.split("|||")
        candidates = fill_mrc_answers(candidates.split(","), alc)
        question = template.replace(question_placeholder, question).replace(choice_placeholder, candidates)
    elif task_name == "regular":
        question = template.replace(question_placeholder, question)
    return question + answer if add_answer else question

def build_demonstrations(icl_cot_data, task_name, query_template):
    demonstrations = "\n\n".join([construct_example("{}|||{}".format(icl_data['question'], ",".join([str(x["answer"]) for x in icl_data["quant_reason.mrc"]])), icl_data['answer'].replace('[Final solution] ', ''), "", query_template, task_name, True) for icl_data in icl_cot_data])
    return demonstrations

def trim_prediction(prediction, answer_prefix, EOSs):
    for eos in EOSs:
        if eos in prediction:
            if prediction.index(eos) > len(eos):
                # eos is not the starter of the prediction
                prediction = prediction.split(eos)[0]
        if prediction.count(answer_prefix) > 1:
            # there is more than one answer prefixes in the response
            prediction = answer_prefix.join(prediction.split(answer_prefix)[:2])

    prediction = trim_punctuation(prediction)
    return prediction

def trim_punctuation(text):
    puncs = string.punctuation + "\n"
    while len(text) and text[-1] in puncs:
        text = text.rstrip(text[-1])
        
    while len(text) and text[0] in puncs:
        text = text.lstrip(text[0])
        
    if len(text) and text.endswith(" "):
        text = text.rstrip()

    if len(text) and text.startswith(" "):
        text = text.lstrip()
        
    return text

def recover_choices(input_str: str, n_choices: int, choice_pool: list, q_suffix: str, combined: bool = False):
    options = input_str.split(q_suffix)[-1].strip() if len(q_suffix) else input_str.strip()
    
    prefixs_to_remove = ['Answer Choices:']
    suffixs_to_remove = ['Let\'s']
    
    for prefix_to_remove in prefixs_to_remove:
        if prefix_to_remove in options:
            options = options[options.index(prefix_to_remove)+len(prefix_to_remove):]
    
    for suffix_to_remove in suffixs_to_remove:
        if suffix_to_remove in options:
            options = options[:options.index(suffix_to_remove)]

    if combined: return options

    choice_templates = ["{}.", "({})", "{})", "{} )"]

    indexes = None
    
    for template in choice_templates:
        if any([template.format(choice_pool[i]) in options for i in range(n_choices)]):
            indexes = [options.find(template.format(choice_pool[i])) for i in range(n_choices)]
            break
                              
    choices = []

    for i in range(len(indexes) - 1):
        choices.append(options[indexes[i]+3:indexes[i+1]].strip())
    choices.append(options[indexes[i+1]+3:].strip())

    choices = [choice.split("\n")[0] if "\n" in choice else choice for choice in choices]

    return choices

def locate_choice(text, candidates_pool, choice_templates):
    pred_choice = None
    cand_mention = []
    for choice_template in choice_templates:
        for cand in candidates_pool:
            choice_template_cand = choice_template.replace('[OPTION]', cand)
            choice_template_cand_lower = choice_template_cand.lower()
            if text.startswith(choice_template_cand) or text.startswith(choice_template_cand_lower):
                # The text needs to be parsed with answer prefix first.
                return cand
            
            if choice_template_cand in text:
                cand_mention += [(cand, m.start(), m.end()) for m in re.finditer(choice_template_cand, text)]
                
    if len(cand_mention):
        # sorted the mention by increasing the start index
        cand_mention_sorted = sorted(cand_mention, key=lambda x:x[1])
        return cand_mention_sorted[-1][0]
    return pred_choice

def elastic_match(prediction, reference, choice_templates, remove_punc=True):
    prediction = prediction.lstrip()
    reference = trim_punctuation(reference)
    
    if remove_punc:
        prediction = trim_punctuation(prediction)
        if not len(prediction): return 0
    
    if prediction.startswith(reference):
        return 1
    
    pred_choice = locate_choice(prediction.lower(), "abcd", choice_templates)
    if pred_choice == reference.lower():
        return 1
    
    return 0

def response_soft_match(prediction, golden_ans, candidates_pool, choice_templates):
    pred_choice = locate_choice(prediction, candidates_pool, choice_templates)
    soft_match = int(pred_choice == golden_ans) if pred_choice else elastic_match(prediction.lower(), golden_ans.lower(), choice_templates)

    if soft_match:
        # find the correct answer
        pred_ans = golden_ans
    elif pred_choice:
        # find another answer
        pred_ans = pred_choice
    else:
        pred_ans = trim_punctuation(prediction)
        
    return soft_match, pred_ans

def response_soft_match_full(question, prediction, response, golden_ans, question_suffix, 
                             n_cands, candidates_pool, choice_templates, answer_extract_patterns):
    # prediction is the preliminary parsing result of response
    meta_info = {}
    pred_separators = ["\n", "."]
    null_prediction = not prediction or len(prediction) == 0
    try:
        choices = recover_choices(question, n_cands, candidates_pool, question_suffix)
    except Exception as err:
        print("recover choice err: {}".format(err))
        pdb.set_trace()
    if null_prediction:
        meta_info["null_pred"] = 1
        choice_appeared = [c for c in choices if c in response]
        if not len(choice_appeared):
            choice_letter_math = None
            for answer_extract_pattern in answer_extract_patterns:
                choice_letter_math = re.search(answer_extract_pattern, response)
                if choice_letter_math:
                    letter_match = choice_letter_math.group(1) == golden_ans
                    soft_match = letter_match
                    pred_ans = choice_letter_math.group(1)
                    meta_info["letter_match"] = 1
                    break
            if not choice_letter_math:
                soft_match = 0
                pred_ans = ""
        elif len(choice_appeared) == 1:
            soft_match = int(candidates_pool[choices.index(choice_appeared[0])] == golden_ans)
            pred_ans = candidates_pool[choices.index(choice_appeared[0])]
        else:
            pred_choice = locate_choice(response.lower(), candidates_pool.lower(), choice_templates)

            if pred_choice:
                soft_match = int(pred_choice.upper() == golden_ans)
                pred_ans = pred_choice.upper()
            else:
                soft_match = 0
                pred_ans = response
    else:
        if len(prediction) < 10: # 10
            soft_match, pred_ans = response_soft_match(prediction, golden_ans, candidates_pool, choice_templates)
        else:
            soft_match, pred_ans = response_soft_match(response, golden_ans, candidates_pool, choice_templates)

    all_cand_ans = candidates_pool
    gold_ans_info = choices[all_cand_ans.index(golden_ans)]
        
    if pred_ans:
        pred_ans = pred_ans.upper().rstrip()
        for pred_separator in pred_separators:
            if pred_separator in pred_ans:
                pred_ans = pred_ans.split(pred_separator)[0]
                pred_ans = pred_ans.rstrip()

        if not len(pred_ans):
            soft_match = 0
            return soft_match, pred_ans, meta_info   

        if len(pred_ans) == 1 and pred_ans in alc:  
            # o/w pred_ans is None or non-letter text
            cand_index = alc.index(pred_ans)
            if cand_index < len(choices):
                soft_match = int(pred_ans.lower() == golden_ans.lower())
            else:
                soft_match = 0

            return soft_match, pred_ans, meta_info

    return soft_match, pred_ans, meta_info

def extract_answer(result, answer_pattern):
    ans = None
    if re.match(answer_pattern, result):
        ans = re.match(answer_pattern, result).groups()[1].lstrip().replace(",", "").split("\n")[0]
        ans = trim_punctuation(ans)
    return ans