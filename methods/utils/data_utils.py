import json
import datasets
import random
import transformers


class TokenizerWrapper:
    def __init__(self, input_ids):
        self.input_ids = input_ids


def read_jsonl(file_name):
    with open(file_name, mode='r') as reader:
        data = json.load(reader)
    return data


def get_reasoning_data(name, nsamples, seed, seqlen, tokenizer, model, eval_mode=False):
    model_name = model.split("/")[-1]
    if name == "reasoning-numina-math-1.5":
        dataset_name = "NuminaMath-1.5"
    elif name == "reasoning-math-500":
        dataset_name = "MATH-500"
    elif name == "reasoning-livecodebench":
        dataset_name = "LiveCodeBench"
    elif name == "reasoning-gpqa":
        dataset_name = "GPQA-Diamond"
    else:
        raise NotImplementedError

    traindata = read_jsonl(f"./datasets/gen_data/{model_name}/{dataset_name}.jsonl")
    traintext = ''
    for item in traindata:
        traintext += item['full_prompt'] + item['generated_text'][0] + "\n\n"
        
    trainenc = tokenizer(traintext, return_tensors='pt')    
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader


def get_wikitext2(nsamples, seed, seqlen, tokenizer, eval_mode=False):
    if eval_mode:
        testdata = datasets.load_dataset('./datasets/wikitext', 'wikitext-2-raw-v1', split='test', trust_remote_code=True)
        testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')
        return testenc
    else:
        traindata = datasets.load_dataset('./datasets/wikitext', 'wikitext-2-raw-v1', split='train')
        traindata = traindata.filter(lambda x: len(x) > 0)
        traindata = traindata.map(lambda x : {'text': x['text'].strip()})
        trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')    
        # random.seed(seed)
        trainloader = []
        for _ in range(nsamples):
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))
        return trainloader


def get_c4_new(nsamples, seed, seqlen, tokenizer, eval_mode=False):
    if eval_mode:
        valdata = datasets.load_dataset(
        './datasets/allenai/c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation')
        valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
        valenc = valenc.input_ids[:, :(256 * seqlen)]
        valenc = TokenizerWrapper(valenc)
        return valenc
    else:
        traindata = datasets.load_dataset(
            './datasets/allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train')
        # random.seed(seed)
        trainloader = []
        for _ in range(nsamples):
            while True:
                i = random.randint(0, len(traindata) - 1)
                trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
                if trainenc.input_ids.shape[1] >= seqlen:
                    break
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))
        return trainloader


def get_ptb_new(nsamples, seed, seqlen, tokenizer, eval_mode=False):
    if eval_mode:
        testdata = datasets.load_dataset('./datasets/ptb_text_only', 'penn_treebank', split='test')
        testenc = tokenizer(" ".join(testdata['sentence']), return_tensors='pt')
        return testenc
    else:
        traindata = datasets.load_dataset('./datasets/ptb_text_only', 'penn_treebank', split='train')
        trainenc = tokenizer(" ".join(traindata['sentence']), return_tensors='pt')
        # random.seed(seed)
        trainloader = []
        for _ in range(nsamples):
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))
        return trainloader


def get_pile(nsamples, seed, seqlen, tokenizer):
    traindata = datasets.load_dataset("./datasets/pile-val-backup", split="validation")
    trainenc = tokenizer("\n\n".join(traindata['text'][:1000]), return_tensors='pt')
    # random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader


def get_loaders(
    name, nsamples=128, seed=0, seqlen=2048, model='', hf_token=None, eval_mode=False
):
    transformers.set_seed(seed)
    if hf_token is None:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model)
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model, use_auth_token=hf_token)
    if 'wikitext2' in name:
        dataset = get_wikitext2(nsamples, seed, seqlen, tokenizer, eval_mode)
    elif 'ptb' in name:
        dataset = get_ptb_new(nsamples, seed, seqlen, tokenizer, eval_mode)
    elif 'c4' in name:
        dataset = get_c4_new(nsamples, seed, seqlen, tokenizer, eval_mode)
    elif 'pile' in name:
        dataset = get_pile(nsamples, seed, seqlen, tokenizer)
    elif name.startswith('reasoning'):
        dataset = get_reasoning_data(name, nsamples, seed, seqlen, tokenizer, model, eval_mode)
    if 'c4' in name and eval_mode:
        dataset = dataset.input_ids
        dataset = TokenizerWrapper(dataset)
    return dataset