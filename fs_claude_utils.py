from argparse import ArgumentParser
import json
import re
from anthropic import Anthropic, APIConnectionError, RateLimitError, APIStatusError
import backoff
from tqdm import tqdm
from fs_chatgpt_utils import sample_episode, filter_large_episode
from fs_ner_utils import CDEBertTokenizer, Sentence


class UniNERExample:
    system_prompt: str = 'You are a helpful information extraction system.'
    max_tokens: int = 512
    temperature: float = 0
    template = 'Given a passage, your task is to extract all entities and identify their entity types. The output should be in a list of tuples of the following format: [("entity 1", "type of entity 1"), ... ].\n\nPassage: {input_passage}'
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
        try:
            start, end = response.find('['), response.rfind(']') + 1
            assert start < end, response
            response = response[start:end]
        except Exception:
            # print(response)
            return None
        
        sent = Sentence(self.text, id=f"example#{self.example_id}")
        
        spans = []
        try:
            spans.extend(json.loads(response))
        except Exception:
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
            "role": "user",
            "content": self.template.format(input_passage=self.text)
        }]
        request_json = {
            "model": "claude-3-haiku-20240307",
            "messages": conversation,
            "system": self.system_prompt, 
            "max_tokens": self.max_tokens,  # The maximum number of tokens (words) in the generated response
            "temperature": self.temperature,
            # Sampling temperature (higher values make the output more random,
            # lower values make it more focused)
        }
        # json has null value
        assert self.example_id is not None, 'we need id to match dataset idx'
        return json.dumps((self.toJson(), request_json))


def run_claude_api(requests_filepath, save_filepath):
    client = Anthropic(
        api_key="[your api key]" ,
        max_retries=0, # default is 2
    )

    @backoff.on_exception(backoff.expo, (APIConnectionError, APIStatusError), max_tries=60, max_value=1800)
    def make_request(request):
        try:
            message = client.messages.create(
                max_tokens=request["max_tokens"],
                temperature=request["temperature"],
                system=request["system"],
                messages=request["messages"],
                model=request["model"],
            )
            return message
        except APIConnectionError as e:
            print("The server could not be reached")
            print(e.__cause__)  # an underlying Exception, likely raised within httpx.
            raise e
        except RateLimitError as e:
            print("A 429 status code was received; we should back off a bit.")
            raise e
        except APIStatusError as e:
            print("Another non-200-range status code was received")
            print(e.status_code)
            print(e.response)
            raise e

    with open(requests_filepath) as fin, open(save_filepath, 'w') as fout:
        for line in tqdm(fin):
            payload, request = json.loads(line.strip())

            try:
                message = make_request(request)
            except Exception as e:
                print(f"Error occurred: {e}")
                continue
            
            res_data = {
                    'payload': payload,
                    'request': request,
                    'response': message.to_dict()
                }
            fout.write(json.dumps(res_data)+'\n')

      
def parse_result(result_fn: str, sent_fn:str, tokenizer: CDEBertTokenizer):
    samples = []
    with open(result_fn) as f:
        for idx, line in enumerate(f):
            result = json.loads(line.strip())
            example = UniNERExample.fromJson(result['payload'])
            completion_tokens = result['response']['usage']['output_tokens']
            message = result['response']['content'][0]['text'].strip()
            # print(example.example_id, completion_tokens)
            sent = example.parseResult(message)
            if sent:
                samples.append(sent)
    
    data = []
    splits = 0
    for sent in tqdm(samples):
        try:
            texts, offsets = tokenizer.word_tokenize(sent.text, False)
            sent.add_token_offset(offsets)
            splits += sent.split_at_char_offset()
            sent.convert_entity_char_offset(mode="strict", nooverlap=True)
            data.append(sent)
        except (RuntimeError, AssertionError):
            continue

    with open(sent_fn, 'w') as f:
        for sent in data:
            f.write(sent.to_json()+'\n')


if __name__ == '__main__':
    from fs_ner_utils import add_fewshot_specific_args, add_train_specific_args, init_argparser
    from fs_chatgpt_utils import add_uniner_specific_args
    from itertools import islice

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
            for i, line in enumerate(islice(f, 10000, 100000)):
                para = json.loads(line.strip())
                samples.append(UniNERExample(para, example_id=i))
            
        request_fn = f'claude/request.jsonl'
        with open(request_fn, 'w') as f:
            for sample in samples:
                f.write(sample.toRequest() + '\n')

    if args.call_api:
        request_fn = 'claude/request.jsonl'
        result_fn = 'claude/result.jsonl'
        run_claude_api(request_fn, result_fn)

    if args.parse_result:
        result_fn = 'claude/result.jsonl'
        sent_fn = 'claude/sentence.jsonl'
        # only do word tokenize so does not care what backbone is used
        tokenizer = CDEBertTokenizer(args.backbone)
        parse_result(result_fn, sent_fn, tokenizer)

    if args.sample_episode:
        sent_fn = f'claude/sentence.jsonl'
        sample_episode(sent_fn, args.mode, 5000, args.UniNway, args.UniKshot, args.UniQtest,
                       args.UniTopK, args.UniAlpha,
                       just_sample=False, output_dir="claude")

    if args.filter_episode:
        filter_large_episode(args, 25000, output_dir="claude")
