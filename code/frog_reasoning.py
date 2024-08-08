import os
from os.path import join
import re
import json
import numpy as np
from string import ascii_uppercase as alc
from utils import dispatch_openai_requests_advanced, \
                  get_text_from_openai_response, json_reader, fill_mrc_answers, \
                  build_demonstrations, get_icl_data, trim_prediction, trim_punctuation, \
                  response_soft_match_full, extract_answer
from config import *
import aiolimiter
import asyncio
from openai import AsyncOpenAI
import argparse
from vllm import LLM, SamplingParams
from datasets import load_dataset

def quant_reasoning(task_name, instruction, num_icl_example, num_data, requests_per_minute, arguments):
    answer_prefix = arguments["answer_prefix"]
    EOSs = ["Question:\n"]
    all_token_consumption = {}

    query_template = PROMPT_INPUT_FOR_FROG
    
    icl_cot_data = get_icl_data(arguments['frog_task'], num_icl_example)
        
    reasoning_data_full = load_dataset("GAIR/FRoG", arguments["frog_task"], arguments["split"])

    reasoning_data = reasoning_data_full[:num_data]

    demonstrations = build_demonstrations(icl_cot_data, task_name, query_template)

    query_prompts = []
    all_questions = []

    for d in reasoning_data:
        answer_key = "answer" if "answer" in d else "gold_ans"

        # eliminate the original candidates from question
        question = d["raw_question"]
        question_prompt = d['question']
        question_prompt = query_template.replace(' [CHOICES]', "").replace('[QUESTION]', question_prompt)

        query_prompt = instruction + demonstrations + "\n" + question_prompt
        all_questions.append(question)
        query_prompts.append(query_prompt)

    if arguments["model_id"].startswith("gpt"):
        limiter = aiolimiter.AsyncLimiter(requests_per_minute)
        client = AsyncOpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base
            )
        responses = asyncio.run(
                dispatch_openai_requests_advanced(
                    messages_list=[
                        [{"role": "user", "content": x}] for x in query_prompts
                    ],
                    client=client,
                    model=arguments["model_id"],
                    temperature=sampling_parameters['temperature'],
                    max_tokens=sampling_parameters['max_tokens'],
                    top_p=sampling_parameters['top_p'],
                    stop_tokens=sampling_parameters["stop"],
                    limiter=limiter
                )
            )
        use_vllm = False
    else:
        # vllm version
        sampling_params = SamplingParams(temperature=sampling_parameters['temperature'],
                                         max_tokens=sampling_parameters['max_tokens'],
                                         top_p=sampling_parameters['top_p'],
                                         stop=stop_tokens
                                         )
        llm = LLM(model=arguments["model_path"], trust_remote_code=True, tensor_parallel_size=arguments["num_gpu"])
        responses = llm.generate(query_prompts, sampling_params)
        use_vllm = True

    all_preds = []
    acc = 0
    num_null_ans = 0
    err = 0
    answer_extract_pattern = arguments["answer_pattern"]
    n_cands = 5 if arguments['frog_task'] == "regular" or arguments['frog_task'].startswith("quant_orig") else 4 
    question_suffix = "?" if arguments['frog_task'] == "regular" else "from the following choices"
    candidates_pool = alc[:n_cands]

    for query_id in range(len(query_prompts)):
        try:
            response_text, response_token_usage = get_text_from_openai_response(responses[query_id], return_usage=True, use_vllm=use_vllm)
        except:
            err += 1
            print("Err: {}".format(query_id))
            continue
        pred_ans_text = trim_prediction(response_text, answer_prefix, EOSs)
        step_sol_info = pred_ans_text.split(ans_starter)

        if len(step_sol_info) > 1:
            pred_ans = step_sol_info[-1]
            reasoning_steps = ans_starter.join(step_sol_info[:2])       
        else:
            reasoning_steps, pred_ans = step_sol_info[0], None

        if pred_ans:
            pred_ans = trim_punctuation(pred_ans)
            pred_ans = pred_ans.rstrip("\n")

        if answer_prefix in reasoning_data[query_id][answer_key]:
            golden_ans_info = reasoning_data[query_id][answer_key].split(answer_prefix)
            golden_ans = golden_ans_info[1] if len(golden_ans_info) == 2 else golden_ans_info
        else:
            golden_ans = reasoning_data[query_id][answer_key]

        golden_ans = trim_punctuation(golden_ans)

        pred_ans_extracted = extract_answer(pred_ans_text, answer_extract_pattern)

        if pred_ans_extracted:
            ans_correct = int(golden_ans.lower() == pred_ans_extracted.lower() or pred_ans_extracted.lower().startswith(golden_ans))
        else:
            question = reasoning_data[query_id]['question']
            soft_match, pred_ans, meta_info = response_soft_match_full(question, reasoning_steps, pred_ans_text, golden_ans, question_suffix, 
                                                                        n_cands, candidates_pool, choice_templates, answer_extract_patterns)
            ans_correct = soft_match

        pred_summary = {
            "question": all_questions[query_id],
            "task_question": reasoning_data[query_id]['question'],
            "gold_ans": golden_ans,
            "pred_reasoning": reasoning_steps,
            "pred_ans": pred_ans,
            "response": response_text,
            "correct": ans_correct,
            "model_id": arguments["model_id"]
            }

        all_preds.append(pred_summary)

        if pred_ans is not None:
            acc += pred_summary['correct']
        else:
            num_null_ans += 1

        for token_key, token_info in response_token_usage.items():
            if token_key not in all_token_consumption:
                all_token_consumption[token_key] = []
            all_token_consumption[token_key].append(token_info)
    

    info = "========== Token Summary ==========\n"
    info += "Err rate: {}\n".format(round(err / (query_id + 1)), 3)
    for key, value in all_token_consumption.items():
        try:
            info += "{}: {}\n".format(key, round(np.mean(value), 2))
        except:
            info += "{}: None\n".format(key)
    info += "Acc: {}\nNull: {}\n".format(round(acc / len(query_prompts), 3),
                                     round(num_null_ans / len(query_prompts), 3))
    print(info)

    if arguments["save_outputs"]:
        arguments["save_path"] = arguments["save_path"].replace("[NUM_SAMPLE]", str(len(query_prompts)))
        with open(arguments["save_path"], "w") as fp:
            for pred in all_preds:
                fp.write(json.dumps(pred) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--shots", default=0, type=int, help="number of in-context learning example used.")
    parser.add_argument("--num_samples", default=0, type=int, help="number of samples used for evaluation")
    parser.add_argument("--task", required=True, type=str)
    parser.add_argument("--split", type=str)
    parser.add_argument("--model_path", type=str, help="backbone model used.")
    parser.add_argument("--model_name", required=True, type=str, help="backbone model used.")
    parser.add_argument("--task_category", type=str, default="quant_reason_mrc")
    parser.add_argument("--request_per_minute", type=int, default=10, help="request per minutes for OpenAI access.")
    parser.add_argument("--save_dir", type=str, default="../result/quant_reason")
    parser.add_argument("--save_outputs", type=int, default=0)
    parser.add_argument("--answer_prefix", type=str, default="####", help="The prefix that identifies the answer.")
    parser.add_argument("--num_gpu", type=int, default=1, help="number of gpu device used for parallization.")
    args = parser.parse_args()

    dataset = args.dataset
    num_icl_example = args.shots
    num_samples = args.num_samples
    requests_per_minute = args.request_per_minute
    save_dir = args.save_dir
    save_dir_model = join(save_dir, args.model_name)

    frog_task = args.task
    task_name = args.task_category

    arguments = {}
    arguments["model_id"] = args.model_name
    arguments["model_path"] = args.model_path
    arguments["num_gpu"] = args.num_gpu
    arguments["answer_prefix"] = args.answer_prefix
    arguments["split"] = args.split
    arguments["frog_task"] = frog_task
    arguments["save_outputs"] = args.save_outputs
    arguments["answer_pattern"] = answer_pattern = "([^#]*){}([^#]*)".format(arguments["answer_prefix"])

    if not os.path.exists(save_dir_model):
        os.makedirs(save_dir_model)

    save_file = "task={}.num_icl={}.num_sample=[NUM_SAMPLE].model={}.json".format(frog_task, num_icl_example, 
                                                                                   args.model_name)      

    save_dir = join(save_dir, args.model_name)
    arguments["save_path"] = join(save_dir, save_file)
    
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    task_instruction = Instruction_frog if frog_reasoning_add_instruction else ""

    quant_reasoning(dataset, task_name, task_instruction, num_icl_example, num_samples, requests_per_minute, arguments)