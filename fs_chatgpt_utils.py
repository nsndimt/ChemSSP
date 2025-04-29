from argparse import ArgumentParser
import asyncio
from copy import deepcopy
from dataclasses import dataclass
import json
import logging
import os
import random
import re
import time
import aiohttp
import numpy as np
import pandas as pd
import pulp
from collections import Counter, defaultdict
from typing import Iterable, List, Optional

import tiktoken
import torch
from tqdm import tqdm
from fs_ner_utils import Episode, EpisodeDataset, Sentence, CDEBertTokenizer, episode_dataset_collate

class UniNERExample:
    system_prompt: str = 'You are a helpful information extraction system.'
    max_tokens: int = 256
    temperature: float = 0
    template = 'Given a passage, your task is to extract all entities and identify their entity types. The output should be in a list of tuples of the following format: [("entity 1", "type of entity 1"), ... ]. Passage: {input_passage}'
    para_args = ("doi", "journal", "title", "section", "merged_section", "merged_section", "text", "offset_start", "offset_end")
    json_args = ("example_id", "response")

    def __init__(self, para, example_id = None, response = None) -> None:
        for kw in self.para_args:
            setattr(self, kw, para[kw])
        self.example_id = example_id
        self.response = response

    def toJson(self):
        return {kw: getattr(self, kw) for kw in self.para_args + self.json_args}
    
    @classmethod
    def fromJson(cls, jsonDict):
        para = {kw: jsonDict[kw] for kw in cls.para_args}
        init_args = {kw: jsonDict[kw] for kw in cls.json_args}
        return cls(para, **init_args)
    
    def parseResult(self, response):
        self.response = response
        spans = []
        try:
            assert response[0] == '[' and response[-1] == ']', response
        except AssertionError:
            return None
        sent = Sentence(self.text, id=f"example#{self.example_id}")
        for re_match in re.finditer(r"(\([\'\"].+?[\'\"]\)(,\s)?)", response[1:-1]):
            pairs = re_match.group(0).strip()
            assert pairs[0] == '(' and (pairs[-1] == ')' or pairs[-2:] == '),'), pairs
            try:
                txt, label = re.findall(r"(\'.+?\')|(\".+?\")", pairs)
                txt, label = txt[0][1:-1] if txt[0] != '' else txt[1][1:-1], label[0][1:-1] if label[0] != '' else label[1][1:-1]
                offsets = [(i.start(), i.end()) for i in re.finditer(re.escape(txt), self.text)] 
                assert len(offsets) > 0, (txt, self.text)
                label = label.lower()
                spans.extend([(start, end, txt, label) for start, end in offsets])
            except:
                continue
        
        try:
            sent.add_entity_char_offset(spans, nooverlap=True)
        except RuntimeError:
            return None
        
        return sent
    
    def toRequest(self):
        conversation = [{
            "role": "system",
            "content": self.system_prompt
        }, {
            "role": "user",
            "content": self.template.format(input_passage=self.text)
        }]
        request_json = {
            "model": "gpt-3.5-turbo",
            "messages": conversation,
            "max_tokens": self.max_tokens,  # The maximum number of tokens (words) in the generated response
            "n": 1,  # The number of completions to generate for the given prompt
            "stop": None,  # String or sequence of strings to stop the text generation when reached
            "temperature": self.temperature,
            # Sampling temperature (higher values make the output more random,
            # lower values make it more focused)
        }
        # json has null value
        assert self.example_id is not None, 'we need id to match dataset idx'
        return json.dumps((self.toJson(), request_json))


def api_endpoint_from_url(request_url):
    """Extract the API endpoint from the request URL."""
    match = re.search('^https://[^/]+/v\\d+/(.+)$', request_url)
    return match[1]


def num_tokens_consumed_from_request(
    request_json: dict,
    api_endpoint: str,
    token_encoding_name: str,
):
    """Count the number of tokens in the request. Only supports completion and embedding requests."""
    encoding = tiktoken.get_encoding(token_encoding_name)
    # if completions request, tokens = prompt + n * max_tokens
    if api_endpoint.endswith("completions"):
        max_tokens = request_json.get("max_tokens", 15)
        n = request_json.get("n", 1)
        completion_tokens = n * max_tokens

        # chat completions
        if api_endpoint.startswith("chat/"):
            num_tokens = 0
            for message in request_json["messages"]:
                num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
                for key, value in message.items():
                    num_tokens += len(encoding.encode(value))
                    if key == "name":  # if there's a name, the role is omitted
                        num_tokens -= 1  # role is always required and always 1 token
            num_tokens += 2  # every reply is primed with <im_start>assistant
            return num_tokens + completion_tokens
        # normal completions
        else:
            prompt = request_json["prompt"]
            if isinstance(prompt, str):  # single prompt
                prompt_tokens = len(encoding.encode(prompt))
                num_tokens = prompt_tokens + completion_tokens
                return num_tokens
            elif isinstance(prompt, list):  # multiple prompts
                prompt_tokens = sum([len(encoding.encode(p)) for p in prompt])
                num_tokens = prompt_tokens + completion_tokens * len(prompt)
                return num_tokens
            else:
                raise TypeError('Expecting either string or list of strings for "prompt" field in completion request')
    # if embeddings request, tokens = input tokens
    elif api_endpoint == "embeddings":
        input = request_json["input"]
        if isinstance(input, str):  # single input
            num_tokens = len(encoding.encode(input))
            return num_tokens
        elif isinstance(input, list):  # multiple inputs
            num_tokens = sum([len(encoding.encode(i)) for i in input])
            return num_tokens
        else:
            raise TypeError('Expecting either string or list of strings for "inputs" field in embedding request')
    # more logic needed to support other API calls (e.g., edits, inserts, DALL-E)
    else:
        raise NotImplementedError(f'API endpoint "{api_endpoint}" not implemented in this script')


def task_id_generator_function():
    """Generate integers 0, 1, 2, and so on."""
    task_id = 0
    while True:
        yield task_id
        task_id += 1


def append_to_jsonl(data, filename: str) -> None:
    """Append a json payload to the end of a jsonl file."""
    json_string = json.dumps(data)
    with open(filename, "a") as f:
        f.write(json_string + "\n")

@dataclass
class StatusTracker:
    """Stores metadata about the script's progress. Only one instance is created."""

    num_tasks_started: int = 0
    num_tasks_in_progress: int = 0  # script ends when this reaches 0
    num_tasks_succeeded: int = 0
    num_tasks_failed: int = 0
    num_rate_limit_errors: int = 0
    num_api_errors: int = 0  # excluding rate limit errors, counted above
    num_other_errors: int = 0
    time_of_last_rate_limit_error: int = 0  # used to cool off after hitting rate limits


@dataclass
class APIRequest:
    """Stores an API request's inputs, outputs, and other metadata. Contains a method to make an API call."""

    task_id: int
    payload_json: Optional[dict]
    request_json: dict
    token_consumption: int
    attempts_left: int
    result = []

    async def call_api(
        self,
        request_url: str,
        request_header: dict,
        retry_queue: asyncio.Queue,
        save_filepath: str,
        status_tracker: StatusTracker,
    ):
        """Calls the OpenAI API and saves results."""
        logging.info(f"Starting request #{self.task_id}")
        error = None
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url=request_url, headers=request_header, json=self.request_json
                ) as response:
                    response = await response.json()
            if "error" in response:
                logging.warning(
                    f"Request {self.task_id} failed with error {response['error']}"
                )
                status_tracker.num_api_errors += 1
                error = response
                if "Rate limit" in response["error"].get("message", ""):
                    status_tracker.time_of_last_rate_limit_error = time.time()
                    status_tracker.num_rate_limit_errors += 1
                    status_tracker.num_api_errors -= 1  # rate limit errors are counted separately

        except Exception as e:  # catching naked exceptions is bad practice, but in this case we'll log & save them
            logging.warning(f"Request {self.task_id} failed with Exception {e}")
            status_tracker.num_other_errors += 1
            error = e
        if error:
            self.result.append(error)
            if self.attempts_left:
                retry_queue.put_nowait(self)
            else:
                logging.error(f"Request {self.request_json} failed after all attempts. Saving errors: {self.result}")
                append_to_jsonl(({
                    'payload': self.payload_json,
                    'request': self.request_json,
                    'response': [str(e) for e in self.result]
                }), save_filepath)
                status_tracker.num_tasks_in_progress -= 1
                status_tracker.num_tasks_failed += 1
        else:
            append_to_jsonl({
                'payload': self.payload_json,
                'request': self.request_json,
                'response': response
            }, save_filepath)
            status_tracker.num_tasks_in_progress -= 1
            status_tracker.num_tasks_succeeded += 1
            logging.debug(f"Request {self.task_id} saved to {save_filepath}")


async def process_api_requests_from_file(
    requests_filepath: str,
    save_filepath: str,
    request_url: str,
    api_key: str,
    max_requests_per_minute: float,
    max_tokens_per_minute: float,
    token_encoding_name: str,
    max_attempts: int,
    logging_level: int,
):

    """Processes API requests in parallel, throttling to stay under rate limits."""
    # constants
    seconds_to_pause_after_rate_limit_error = 15
    seconds_to_sleep_each_loop = 0.001  # 1 ms limits max throughput to 1,000 requests per second

    # initialize logging
    logging.basicConfig(level=logging_level)
    logging.debug(f"Logging initialized at level {logging_level}")

    # infer API endpoint and construct request header
    api_endpoint = api_endpoint_from_url(request_url)
    request_header = {"Authorization": f"Bearer {api_key}"}

    # initialize trackers
    queue_of_requests_to_retry = asyncio.Queue()
    task_id_generator = task_id_generator_function()  # generates integer IDs of 1, 2, 3, ...
    status_tracker = StatusTracker()  # single instance to track a collection of variables
    next_request = None  # variable to hold the next request to call

    # initialize available capacity counts
    available_request_capacity = max_requests_per_minute
    available_token_capacity = max_tokens_per_minute
    last_update_time = time.time()

    # initialize flags
    file_not_finished = True  # after file is empty, we'll skip reading it
    logging.debug(f"Initialization complete.")

    # initialize file reading
    with open(requests_filepath) as file:
        # `requests` will provide requests one at a time
        requests = file.__iter__()
        logging.debug(f"File opened. Entering main loop")

        while True:
            # get next request (if one is not already waiting for capacity)
            if next_request is None:
                if not queue_of_requests_to_retry.empty():
                    next_request = queue_of_requests_to_retry.get_nowait()
                    logging.debug(f"Retrying request {next_request.task_id}: {next_request}")
                elif file_not_finished:
                    try:
                        # get new request
                        payload_json, request_json = json.loads(next(requests))
                        task_id = next(task_id_generator)
                        next_request = APIRequest(
                            task_id=task_id,
                            payload_json=payload_json,
                            request_json=request_json,
                            token_consumption=num_tokens_consumed_from_request(request_json, api_endpoint, token_encoding_name),
                            attempts_left=max_attempts,
                        )
                        status_tracker.num_tasks_started += 1
                        status_tracker.num_tasks_in_progress += 1
                        logging.debug(f"Reading request {next_request.task_id}: {next_request}")
                    except StopIteration:
                        # if file runs out, set flag to stop reading it
                        logging.debug("Read file exhausted")
                        file_not_finished = False

            # update available capacity
            current_time = time.time()
            seconds_since_update = current_time - last_update_time
            available_request_capacity = min(
                available_request_capacity + max_requests_per_minute * seconds_since_update / 60.0,
                max_requests_per_minute,
            )
            available_token_capacity = min(
                available_token_capacity + max_tokens_per_minute * seconds_since_update / 60.0,
                max_tokens_per_minute,
            )
            last_update_time = current_time

            # if enough capacity available, call API
            if next_request:
                next_request_tokens = next_request.token_consumption
                if (
                    available_request_capacity >= 1
                    and available_token_capacity >= next_request_tokens
                ):
                    # update counters
                    available_request_capacity -= 1
                    available_token_capacity -= next_request_tokens
                    next_request.attempts_left -= 1

                    # call API
                    asyncio.create_task(
                        next_request.call_api(
                            request_url=request_url,
                            request_header=request_header,
                            retry_queue=queue_of_requests_to_retry,
                            save_filepath=save_filepath,
                            status_tracker=status_tracker,
                        )
                    )
                    next_request = None  # reset next_request to empty

            # if all tasks are finished, break
            if status_tracker.num_tasks_in_progress == 0:
                break

            # main loop sleeps briefly so concurrent tasks can run
            await asyncio.sleep(seconds_to_sleep_each_loop)

            # if a rate limit error was hit recently, pause to cool down
            seconds_since_rate_limit_error = (time.time() - status_tracker.time_of_last_rate_limit_error)
            if seconds_since_rate_limit_error < seconds_to_pause_after_rate_limit_error:
                remaining_seconds_to_pause = (seconds_to_pause_after_rate_limit_error - seconds_since_rate_limit_error)
                await asyncio.sleep(remaining_seconds_to_pause)
                # ^e.g., if pause is 15 seconds and final limit was hit 5 seconds ago
                logging.warn(f"Pausing to cool down until {time.ctime(status_tracker.time_of_last_rate_limit_error + seconds_to_pause_after_rate_limit_error)}")

        # after finishing, log final status
        logging.info(f"""Parallel processing complete. Results saved to {save_filepath}""")
        if status_tracker.num_tasks_failed > 0:
            logging.warning(f"{status_tracker.num_tasks_failed} / {status_tracker.num_tasks_started} requests failed. Errors logged to {save_filepath}.")
        if status_tracker.num_rate_limit_errors > 0:
            logging.warning(f"{status_tracker.num_rate_limit_errors} rate limit errors received. Consider running at a lower rate.")

def parallel_run_api(requests_filepath, save_filepath):
    import asyncio
    asyncio.run(
        process_api_requests_from_file(
            requests_filepath=requests_filepath,
            save_filepath=save_filepath,
            request_url="https://api.openai.com/v1/chat/completions",
            api_key="[your api key]",
            max_requests_per_minute=float(1000),
            max_tokens_per_minute=float(100000),
            token_encoding_name="cl100k_base",
            # according to https://github.com/openai/tiktoken/blob/main/tiktoken/model.py
            max_attempts=5,
            logging_level=20,
        )
    )

def parse_result(result_fn: str, sent_fn:str, tokenizer: CDEBertTokenizer):
    samples = []
    with open(result_fn) as f:
        for idx, line in enumerate(f):
            result = json.loads(line.strip())
            example = UniNERExample.fromJson(result['payload'])
            completion_tokens = result['response']['usage']['completion_tokens']
            message = result['response']['choices'][0]['message']['content'].strip()
            # print(example.example_id, completion_tokens)
            sent = example.parseResult(message)
            if sent:
                samples.append(sent)
    
    data = []
    splits = 0
    for sent in samples:
        try:
            texts, offsets = tokenizer.word_tokenize(sent.text, False)
            sent.add_token_offset(offsets)
            splits += sent.split_at_char_offset()
            sent.convert_entity_char_offset(mode="strict", nooverlap=True)
            data.append(sent)
        except RuntimeError:
            continue

    with open(sent_fn, 'w') as f:
        for sent in data:
            f.write(sent.to_json()+'\n')



class EpisodeSampler:

    def __init__(self, N: int, K: int, Q: int, topK: int, alpha: float, loader: Iterable[Sentence]):
        self.N = N
        self.K = K
        self.Q = Q
        self.alpha = alpha
        assert 0 <= alpha <= 1

        self.samples = []
        self.sample_counter = []
        types_counter = Counter()

        for sample in loader:
            self.samples.append(sample)
            self.sample_counter.append(sample.entity_counter)
            types_counter.update(list(sample.entity_counter.keys()))

        self.sample_counter = pd.DataFrame(self.sample_counter).fillna(0).convert_dtypes()
        top_columns, top_cnt = zip(*[(col, cnt) for col, cnt in types_counter.most_common(topK) if cnt >= 2 * self.K])
        self.sample_counter = self.sample_counter.drop([col for col in types_counter if col not in top_columns], axis=1)
        self.types_counter = np.array(top_cnt)
        self.types = top_columns

    def validate(self, support_sample_idxs, support_samples, query_sample_idxs, query_samples, types):
        support_cnt = Counter()
        for sent in support_samples:
            support_cnt += sent.entity_counter

        query_cnt = Counter()
        for sent in query_samples:
            query_cnt += sent.entity_counter

        assert all(support_cnt[c] >= self.K for c in types)
        assert all(query_cnt[c] >= self.K for c in types)
        assert len(set(support_sample_idxs).intersection(set(query_sample_idxs))) == 0

    def solve(self, lines, minimum_occurance):
        model = pulp.LpProblem("FewShot", pulp.LpMinimize)
        lp_vars = []
        class2var = defaultdict(list)
        for idx, entity_cnt in lines.items():
            indi = pulp.LpVariable(f"use_{idx}", cat="Binary")
            lp_vars.append((indi, 1))
            for key, cnt in entity_cnt:
                class2var[key].append((indi, cnt))
        
        model += pulp.lpSum([w * indi for indi, w in lp_vars]), "NumSent"
        
        for key in class2var:
            model += pulp.lpSum([w * indi for indi, w in class2var[key]]) >= minimum_occurance

        model.solve(pulp.PULP_CBC_CMD(msg=0))
        if pulp.LpStatus[model.status] != "Optimal":
            return None

        used_idxs = []
        for indi, w in lp_vars:
            if indi.varValue > 0:
                index = int(indi.name[4:])
                used_idxs.append(index)
        
        total_cost = pulp.value(model.objective)
        return used_idxs
    
    def sample(self, seed):
        random.seed(seed)
        weights = np.power(self.types_counter, self.alpha)
        
        max_iter = 1000
        while max_iter:
            types = random.choices(self.types, weights=weights, k=self.N)
            max_iter -= 1
            if len(set(types)) == self.N:
                print(sorted(types))
                break
        if max_iter == 0:
            return None

        support_cands = {}
        support_cands_cnt = Counter()

        for key in types:
            select = self.sample_counter[key] > 0
            samples = self.sample_counter[select].sample(self.K, replace=False, random_state=seed)
            for index, line in samples.iterrows():
                if index in support_cands:
                    continue
                support_cands[index] = []
                for key in types:
                    cnt = line[key]
                    if cnt > 0:
                        support_cands_cnt[key] += 1
                        support_cands[index].append((key, cnt))

        support_sample_idxs = self.solve(support_cands, self.K)
        if support_sample_idxs is None:
            return None
        
        query_cands = {}
        query_cands_cnt = Counter()

        for key in types:
            select = (self.sample_counter[key] > 0) & (~self.sample_counter.index.isin(support_sample_idxs))
            samples = self.sample_counter[select].sample(self.Q, replace=False, random_state=seed)
            for index, line in samples.iterrows():
                assert index not in support_sample_idxs
                if index in query_cands:
                    continue
                query_cands[index] = []
                for key in types:
                    cnt = line[key]
                    if cnt > 0:
                        query_cands_cnt[key] += 1
                        query_cands[index].append((key, cnt))

        query_sample_idxs = self.solve(query_cands, self.Q)
        if query_sample_idxs is None:
            return None

        query_samples = [mask_other_entity(self.samples[index], types) for index in query_sample_idxs]
        support_samples = [mask_other_entity(self.samples[index], types) for index in support_sample_idxs]

        self.validate(support_sample_idxs, support_samples, query_sample_idxs, query_samples, types)
        episode = Episode(query_samples, support_samples, types)
        assert len(episode.support) > 0 and len(episode.query) > 0, "empty support or query"

        return episode

# return a new copy of sentence with selected types of entity
def mask_other_entity(sent, types):
    new_sent = deepcopy(sent)
    new_sent.add_entity_token_idx([(s, e, l) for s, e, l in sent.entity if l in types])
    new_sent.convert_entity_position(sent.mode)
    return new_sent


def sample_episode(sent_fn, mode, Num_episode, N_way, K_shot, Q_test, topK, alpha, just_sample=True, output_dir=None):
    data = []
    with open(sent_fn) as f:
        for line in f:
            sent = Sentence.from_json(line.strip())
            sent.convert_entity_position(mode)
            data.append(sent)

    print(f"UniNER num of sentence {len(data)}")

    sampler = EpisodeSampler(N_way, K_shot, Q_test, topK, alpha, data)
    pbar = tqdm()
    success_data = []
    support_sizes = []
    pbar.set_description(f"success: 0 fail: 0 total: 0")
    num_iter, iter_MAX = 0, 100000
    assert Num_episode <= iter_MAX
    while len(success_data) < Num_episode and num_iter < iter_MAX:
        ret = sampler.sample(num_iter)
        if ret is not None:
            success_data.append(ret)
            support_sizes.append(len(ret.support))

        num_iter += 1
        pbar.set_description(f"success: {len(success_data)} fail: {num_iter - len(success_data)} total: {num_iter}")

    print(
        f"1% support size {np.percentile(support_sizes, 1)}, "
        f"2% support size {np.percentile(support_sizes, 2)}, "
        f"25% support size {np.percentile(support_sizes, 25)}"
    )
    print(
        f"75% support size {np.percentile(support_sizes, 75)}, "
        f"98% support size {np.percentile(support_sizes, 98)}, "
        f"99% support size {np.percentile(support_sizes, 99)}"
    )

    if not just_sample:
        if output_dir is None:
            output_dir = "episode"
        with open(os.path.join(output_dir, f"UniNER_N{N_way}_K{K_shot}_Q{Q_test}_top_{topK}_alpha_{alpha}.jsonl"), "w") as f:
            f.write(json.dumps(sampler.types) + '\n')
            for episode in success_data:
                f.write(episode.to_json() + "\n")


def load_UniNER_episode(args, training=False, debug_firstk=None):
    mode = args.mode
    
    with open(os.path.join("episode", f"UniNER_N{args.UniNway}_K{args.UniKshot}_Q{args.UniQtest}_top_{args.UniTopK}_alpha_{args.UniAlpha}.jsonl")) as f:
        data = []
        labels = json.loads(f.readline())
        for line in tqdm(f):
            episode = Episode.from_json(line)
            data.append(episode)

            if debug_firstk is not None and len(data) == debug_firstk:
                break

    dataset = EpisodeDataset(data, labels, mode, args)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=episode_dataset_collate,
        batch_size=args.bs if training else args.bs * 2,
        shuffle=training and not args.nan_detect,
        num_workers=2 if not args.nan_detect else 0,
        drop_last=training,
    )
    return dataset, dataloader


def filter_large_episode(args, limit, output_dir=None):
    if output_dir is None:
        output_dir = "episode"
    
    with open(os.path.join(output_dir, f"UniNER_N{args.UniNway}_K{args.UniKshot}_Q{args.UniQtest}_top_{args.UniTopK}_alpha_{args.UniAlpha}.jsonl")) as f:
        data = []
        labels = json.loads(f.readline())
        for line in f:
            episode = Episode.from_json(line)
            data.append(episode)

    dataset = EpisodeDataset(data, labels, args.mode, args)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=episode_dataset_collate,
        batch_size=1,
        num_workers=2,
    )

    filtered = []
    for i, batch in enumerate(dataloader):
        num_seq, seq_len = batch["bert"]["input_id"].size()
        print(num_seq, seq_len, num_seq * seq_len)
        if num_seq * seq_len < limit:
            filtered.append(data[i])

    with open(os.path.join(output_dir, f"UniNER_N{args.UniNway}_K{args.UniKshot}_Q{args.UniQtest}_top_{args.UniTopK}_alpha_{args.UniAlpha}_filtered.jsonl"), "w") as f:
        f.write(json.dumps(labels) + '\n')
        for episode in filtered:
            f.write(episode.to_json() + "\n")


def add_uniner_specific_args(parser: ArgumentParser):
    parser.add_argument("--UniNway", default=5, type=int, help="Nway")
    parser.add_argument("--UniKshot", default=10, type=int, help="Kshot")
    parser.add_argument("--UniQtest", default=10, type=int, help="Qtest")
    parser.add_argument("--UniTopK", default=500, type=int, help="topK")
    parser.add_argument("--UniAlpha", default=0.5, type=float, help="sampling alpha")
    return parser


if __name__ == '__main__':
    from fs_ner_utils import add_fewshot_specific_args, add_train_specific_args, init_argparser
    parser = ArgumentParser()
    parser = init_argparser(parser)
    parser = add_train_specific_args(parser)
    parser = add_fewshot_specific_args(parser)
    parser = add_uniner_specific_args(parser)
    parser.add_argument("--create_task", action='store_true')
    parser.add_argument("--call_api", action='store_true')
    parser.add_argument("--parse_result", action='store_true')
    parser.add_argument("--sample_episode", action='store_true')
    parser.add_argument("--filter_episode", action='store_true')
    args, _ = parser.parse_known_args()

    if args.create_task:
        samples = []
        with open("sampled_filterd_paragraph_chunk.jsonl") as f:
            for i, line in enumerate(f):
                para = json.loads(line.strip())
                samples.append(UniNERExample(para, example_id=i))
            
        request_fn = f'chatGPT/request.jsonl'
        with open(request_fn, 'w') as f:
            for sample in samples:
                f.write(sample.toRequest() + '\n')

    if args.call_api:
        request_fn = 'chatGPT/request.jsonl'
        result_fn = 'chatGPT/result.jsonl'
        parallel_run_api(request_fn, result_fn)

    if args.parse_result:
        result_fn = 'chatGPT/result.jsonl'
        sent_fn = 'chatGPT/sentence.jsonl'
        # only do word tokenize so does not care what backbone is used
        tokenizer = CDEBertTokenizer(args.backbone)
        parse_result(result_fn, sent_fn, tokenizer)

    if args.sample_episode:
        sent_fn = f'chatGPT/sentence.jsonl'
        sample_episode(sent_fn, args.mode, 5000, args.UniNway, args.UniKshot, args.UniQtest, args.UniTopK, args.UniAlpha, just_sample=False)

    if args.filter_episode:
        filter_large_episode(args, 25000)