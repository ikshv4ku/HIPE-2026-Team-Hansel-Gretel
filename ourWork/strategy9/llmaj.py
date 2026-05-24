from collections import defaultdict
import os
import sys
import json
import argparse
from tqdm import tqdm
from dotenv import dotenv_values
import pandas as pd
import requests
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
import re


def clean_special_tokens(text):
    """Remove special tokens from generated text"""
    if not text:
        return ""

    # List of special tokens to remove
    special_tokens = [
        r"<\|END_OF_TURN_TOKEN\|>",
        r"<\|END_RESPONSE\|>",
        r"<\|END_RESPONSE\|><EOS_TOKEN>",
        r"<end_of_turn>",
        r"<turn\|>",
        r"<pad>",
        r"<EOS_TOKEN>",
    ]

    cleaned_text = text
    for token in special_tokens:
        cleaned_text = re.sub(token, "", cleaned_text)

    # Remove extra whitespace
    cleaned_text = " ".join(cleaned_text.split())
    return cleaned_text.strip()


def get_rits_model_list(api_key):
    url = "https://rits.fmaas.res.ibm.com/ritsapi/inferenceinfo"
    response = requests.get(url, headers={"RITS_API_KEY": api_key})
    if response.status_code == 200:
        return {m["model_name"]: m["endpoint"] for m in response.json()}
    else:
        raise Exception(f"Failed getting RITS model list:\n\n{response.text}")


PROMPT = """You are an evaluator. Your task is to compare a Ground-truth Answer and a Prediction to decide if the Prediction correctly answers the given Question. 

Evaluation Rules: 
(1) Correctness: A correct prediction must include all essential information from the Ground-truth Answer. Extra information is allowed if it does not contradict the Ground-truth. If the Prediction states something as a possibility, treat it as a definitive statement. 
(2) Function, Tool Names, and API Calls: If the Ground-truth Answer contains specific function names, tool names, or API calls, the Prediction must contain exactly the same ones. If these are missing, replaced, or altered, the Prediction is incorrect. 
(3) URLs: If the Ground-truth Answer contains specific URLs, the Prediction must contain exactly the same URLs. If these are missing, replaced, or altered, the Prediction is incorrect. 
(4) Language: The answer should be in the same language as the ground-truth answer.

Scoring Rules:
If the Prediction is correct according to the above rules, output <score>[[1]]</score>. If the Prediction is incomplete or incorrect, output <score>[[0]]</score>. 

Output Format:
<explanation>
...
</explanation> 
<score>
...
</score>

First provide reasoning inside <explanation> and </explanation> tags. Then output the score as specified above within <score> and </score> tags. Do not include any extra text outside these tags. 
"""

user_message = """Question: {QUESTION}
Ground-truth Answer: {ANSWER}
Prediction: {ASSIST_ANSWER} 
"""


def llmaj(client, model, question, reference_answer, generated_answer):
    chat_prompt = [
        {"role": "system", "content": PROMPT},
        {
            "role": "user",
            "content": user_message.format(
                QUESTION=question,
                ANSWER=reference_answer,
                ASSIST_ANSWER=generated_answer,
            ),
        },
    ]
    try:
        completion = (
            client.chat.completions.create(
                model=model,
                messages=chat_prompt,
                max_completion_tokens=1024,
                temperature=0,
            )
            .choices[0]
            .message.content
        )

        score, explanation = extract_score_explanation_silt(completion)
    except:
        score = 0
        explanation = "Error"

    return score, explanation


def extract_score_explanation_silt(response):
    pattern_score = re.compile(r"(?<=<score>).*?(\d+).*?(?=<\/score>)", re.DOTALL)
    match_score = pattern_score.search(response)
    if not match_score:
        return None, response
    else:
        return match_score.group(1), response


def calculate_llmaj_score(
    input_file, judge_name, question_col, prediction_col, reference_col
):
    judge_name_to_mname = {}
    judge_name_to_mname["llama70b"] = "meta-llama/llama-3-3-70b-instruct"
    judge_name_to_mname["granite"] = "ibm-granite/granite-3.2-8b-instruct"
    judge_name_to_mname["llama4"] = "meta-llama/llama-4-maverick-17b-128e-instruct-fp8"
    judge_name_to_mname["gpt"] = "openai/gpt-oss-120b"

    input_file = input_file

    config = dotenv_values(".env")

    if "RITS_API_KEY" in config:
        api_key = config["RITS_API_KEY"]

    print("Start processing: ", input_file)

    minfo = get_rits_model_list(api_key)
    mname = judge_name_to_mname[judge_name]
    if judge_name == "llama70b":
        minfo[mname] = (
            "https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/llama-3-3-70b-instruct"
        )
    elif judge_name == "gpt":
        minfo[mname] = (
            "https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/gpt-oss-120b"
        )
    url = f"{minfo[mname]}/v1"
    print("URL: ", url)

    client = OpenAI(
        api_key=api_key,
        base_url=url,
        default_headers={"RITS_API_KEY": api_key},
    )

    with open(input_file) as fp:
        dataset = json.load(fp)

    average_score = 0.0
    for each_instance in tqdm(dataset["dataset"]):
        question = each_instance[question_col]
        reference_answer = each_instance[reference_col]
        generated_answer = clean_special_tokens(each_instance[prediction_col])

        each_instance["llmaj_score"], each_instance["llmaj_response"] = llmaj(
            client, mname, question, reference_answer, generated_answer
        )
        try:
            average_score += float(each_instance["llmaj_score"])
        except:
            average_score += 0.0

    print(f"Average Score: {average_score/len(dataset['dataset'])}")
    dataset[f"{judge_name}_llmaj_score"] = average_score / len(dataset["dataset"])

    return dataset


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-input_file", "--input_file", type=str, help="path of input file"
    )
    parser.add_argument(
        "-judge_name",
        "--judge_name",
        default="llama70b",
        choices=["llama70b", "granite", "mixtral", "llama4", "gpt"],
    )
    parser.add_argument(
        "-question_col",
        "--question_col",
        type=str,
        default="question",
        help="name of col containing question",
    )
    parser.add_argument(
        "-prediction_col",
        "--prediction_col",
        type=str,
        default="predicted_response",
        help="name of col containing predicted response",
    )
    parser.add_argument(
        "-reference_col",
        "--reference_col",
        type=str,
        default="gold_response",
        help="name of col containing gold response",
    )

    args = parser.parse_args()

    data = calculate_llmaj_score(
        args.input_file,
        args.judge_name,
        args.question_col,
        args.prediction_col,
        args.reference_col,
    )

    with open(args.input_file, "w") as outfile:
        json.dump(data, outfile, indent=2)


if __name__ == "__main__":
    main()
