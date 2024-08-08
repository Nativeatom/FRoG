query_template = "[QUESTION]"

model_id = "gpt-4"

openai_api_key = ""
openai_api_base = ""

stop_tokens = ["Question:", "\n\n\n", "<eos>", "<|eot_id|>"]
answer_extract_patterns = [r"answer(?: to .+)? is ([A-D])\.", r"answer(?: to .+)? is ([a-d])\."]
choice_templates = ["option [OPTION]", "choice [OPTION]",
                    "([OPTION])", "[OPTION]. ", 
                    "choose [OPTION]", 
                    "answer: [OPTION]",
                    "answer is [OPTION]",
                    "[OPTION]."]

# source: https://platform.openai.com/docs/api-reference/chat/create
sampling_parameters = {
    "max_tokens": 2000,
    "n": 1,
    "top_p": 1,
    "stop": stop_tokens,
    "temperature": 0.9
}

# quantifier substitution
quant2strength = {
    "all": 0.885,
    "most": 0.687,
    "moderate amount": 0.369,
    "some": 0.225,
    "small amount": 0.183,
    "few": 0.074,
    "tiny amount": 0.024,
    "none": 0.004
}

frog_reasoning_add_instruction = True

icl_cot_dir = "../in_context_learning_examples"

ans_starter = "#### "

Instruction_frog = '''
You are an expect in mathematical reasoning and generalized quantifier reasoning. Here you are asked to answer one mathematical question based on real life scenarios with description starting with 'Question:'. For example, the question may describe the driving experience of a person. Your answer will start with 'Answer: let's think step by step'. 

You will also be provided with four possible choices, please select the choice that is closest to your estimation of the answer.

The answer needs to include necessary reasoning steps to demonstrate your thinking procedure, and the final result of your calculation is demonstrated at the end of your answer starting with '####'. 

Here are some examples starting with 'Question: ' for your reference.

'''

Instruction_mathqa_quant_reason_orig = '''
You are an expect in mathematical reasoning and generalized quantifier reasoning. Here you are asked to answer one mathematical question based on real life scenarios with description starting with 'Question:'. For example, the question may describe the driving experience of a person. Your answer will start with 'Answer: let's think step by step'. 

You will also be provided with five possible choices, please select the choice that is closest to your estimation of the answer.

The answer needs to include necessary reasoning steps to demonstrate your thinking procedure, and the final result of your calculation is demonstrated at the end of your answer starting with '####'. 

Here are some examples starting with 'Question: ' for your reference.

'''

question_placeholder = "[QUESTION]"
choice_placeholder = "[CHOICES]"

PROMPT_INPUT_FOR_FROG = "Question:\n{} {}\nAnswer:\nLet's think step by step.\n".format(question_placeholder, choice_placeholder)
