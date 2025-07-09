from __future__ import annotations

import argparse, os, json, re
import deepspeed, torch

from transformers import AutoTokenizer
from accelerate.utils import set_seed

from eagle.train.eagle3.cnets import Model
from eagle.train.eagle3.configs import EConfig
from datasets import load_dataset, concatenate_datasets
from typing import Any, Dict, List

from torch import nn
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

torch.backends.cuda.matmul.allow_tf32 = True
set_seed(0)


def download_and_mix_datasets(cache_dir="./data_cache", train_split_ratio=0.95):
    """Download UltraChat_200k and shareGPT datasets and mix them into a unified format"""
    
    print("Downloading and processing datasets...")
    
    # Download UltraChat_200k dataset
    try:
        ultrachat_ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft", cache_dir=cache_dir)
        print(f"Loaded UltraChat_200k: {len(ultrachat_ds)} examples")
    except Exception as e:
        print(f"Failed to load UltraChat_200k: {e}")
        ultrachat_ds = None
    
    # Download shareGPT dataset (using a reliable variant)
    try:
        sharegpt_ds = load_dataset("anon8231489123/ShareGPT_Vicuna_unfiltered", cache_dir=cache_dir)
        if 'train' in sharegpt_ds:
            sharegpt_ds = sharegpt_ds['train']
        else:
            # If no train split, use the first available split
            sharegpt_ds = list(sharegpt_ds.values())[0]
        print(f"Loaded shareGPT: {len(sharegpt_ds)} examples")
    except Exception as e:
        print(f"Failed to load shareGPT, trying alternative...")
        try:
            # Fallback to RyokoAI shareGPT dataset
            sharegpt_ds = load_dataset("RyokoAI/ShareGPT52K", split="train", cache_dir=cache_dir)
            print(f"Loaded alternative shareGPT: {len(sharegpt_ds)} examples")
        except Exception as e2:
            print(f"Failed to load alternative shareGPT: {e2}")
            sharegpt_ds = None
    
    mixed_conversations = []
    
    # Process UltraChat_200k data
    if ultrachat_ds:
        print("Processing UltraChat_200k...")
        for example in tqdm(ultrachat_ds, desc="Processing UltraChat"):
            try:
                # UltraChat format: {'messages': [{'role': 'user/assistant', 'content': '...'}]}
                messages = example.get('messages', [])
                if len(messages) >= 2:  # Need at least one exchange
                    conversations = []
                    for msg in messages:
                        role = 'human' if msg['role'] == 'user' else 'gpt'
                        conversations.append({
                            'from': role,
                            'value': msg['content']
                        })
                    
                    mixed_conversations.append({
                        'id': f"ultrachat_{len(mixed_conversations)}",
                        'conversations': conversations
                    })
            except Exception as e:
                continue
    
    # Process shareGPT data
    if sharegpt_ds:
        print("Processing shareGPT...")
        for example in tqdm(sharegpt_ds, desc="Processing shareGPT"):
            try:
                # shareGPT can have different formats, try to handle them
                if 'conversations' in example:
                    # Format: {'conversations': [{'from': 'human/gpt', 'value': '...'}]}
                    conversations = example['conversations']
                elif 'data' in example:
                    # Alternative format: {'data': [...]}
                    conversations = []
                    for item in example['data']:
                        if isinstance(item, list) and len(item) == 2:
                            role, content = item
                            role = 'human' if role == 'human' else 'gpt'
                            conversations.append({'from': role, 'value': content})
                else:
                    continue
                
                if len(conversations) >= 2:  # Need at least one exchange
                    mixed_conversations.append({
                        'id': f"sharegpt_{len(mixed_conversations)}",
                        'conversations': conversations
                    })
            except Exception as e:
                continue
    
    print(f"Total mixed conversations: {len(mixed_conversations)}")
    
    # Split into train and test
    split_idx = int(len(mixed_conversations) * train_split_ratio)
    train_data = mixed_conversations[:split_idx]
    test_data = mixed_conversations[split_idx:]
    
    # Save mixed dataset locally
    os.makedirs(cache_dir, exist_ok=True)
    train_path = os.path.join(cache_dir, "mixed_train.jsonl")
    test_path = os.path.join(cache_dir, "mixed_test.jsonl")
    
    with open(train_path, 'w', encoding='utf-8') as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    with open(test_path, 'w', encoding='utf-8') as f:
        for item in test_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"Saved train dataset: {len(train_data)} examples -> {train_path}")
    print(f"Saved test dataset: {len(test_data)} examples -> {test_path}")
    
    return train_path, test_path


def build_dataset_rank(tokenizer, datapath):
  ds = load_dataset('json', data_files=datapath)
  ds = ds['train']
  ds = ds.shuffle(seed=42)
  ds1 = ds
  original_columns1 = ds1.column_names
  num_proc = 8

  def preprocess_function(examples):
    new_examples = {'attention_mask': [], 'input_ids': [], 'loss_mask': []}
    for i in range(len(examples['id'])):
      messages = [
        {
          'role': 'system',
          'content': "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.",
        }
      ]
      convroles = ['user', 'assistant']
      roles = {'human': 'user', 'gpt': 'assistant'}
      source = examples['conversations'][i]
      if not source:
        continue
      if roles[source[0]['from']] != 'user':
        # Skip the first one if it is not from human
        source = source[1:]
      for j, sentence in enumerate(source):
        role = roles[sentence['from']]
        assert role == convroles[j % 2], f'{i}'
        # if sentence["from"]=="gpt":
        #     sentence["value"]=" "+sentence["value"]
        messages.append({'role': role, 'content': sentence['value']})
      conversation = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

      if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.unk_token_id

      input_ids = tokenizer(conversation, return_tensors='pt', max_length=2048, add_special_tokens=False).input_ids[0]
      loss_mask = torch.ones_like(input_ids)
      # print(i)

      sep = '<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'

      total_len = len(input_ids)

      sep2 = '<|eot_id|><|start_header_id|>user<|end_header_id|>'
      turns = conversation.split(sep2)

      turns[1] = turns[0] + sep2 + turns[1]
      turns = turns[1:]

      cur_len = 1
      loss_mask[:cur_len] = 0
      for i, turn in enumerate(turns):
        if turn == '':
          break
        turn_len = len(tokenizer(turn).input_ids)

        parts = turn.split(sep)
        if len(parts) != 2:
          break
        parts[0] += sep
        # "-2" is hardcoded for the Llama tokenizer to make the offset correct.
        instruction_len = len(tokenizer(parts[0]).input_ids) - 1

        # Ignore the user instructions
        if i == 0:
          loss_mask[cur_len : cur_len + instruction_len - 2] = 0
        else:
          loss_mask[cur_len - 3 : cur_len + instruction_len + 1] = 0
        cur_len += turn_len
        if i != 0:
          cur_len += 3
        # cur_len+=2

        # if i != 0 and not tokenizer.legacy:
        #     # The legacy and non-legacy modes handle special tokens differently
        #     cur_len -= 1

      loss_mask[cur_len:] = 0
      attention_mask = torch.ones_like(loss_mask)

      # new_examples["conversation"].append(conversation)
      new_examples['input_ids'].append(input_ids[None, :])
      new_examples['loss_mask'].append(loss_mask[None, :])
      new_examples['attention_mask'].append(attention_mask[None, :])

    return new_examples

  ds1 = ds1.map(
    preprocess_function, batched=True, num_proc=num_proc, remove_columns=original_columns1, load_from_cache_file=False
  )

  ds1.set_format(type='torch')
  return ds1


class DataCollatorWithPadding:
  def paddingtensor(self, intensors, N):
    B, n, S = intensors.shape
    # padding_tensor = torch.zeros(B, N - n, S,dtype=intensors.dtype)
    padding_tensor = torch.zeros(B, N - n, S, dtype=intensors.dtype)
    outtensors = torch.cat((intensors, padding_tensor), dim=1)
    return outtensors

  def paddingtensor2D(self, intensors, N):
    B, n = intensors.shape
    padding_tensor = torch.zeros(B, N - n, dtype=intensors.dtype)
    outtensors = torch.cat((intensors, padding_tensor), dim=1)
    return outtensors

  def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
    max_length = max(item['input_ids'].shape[1] for item in features)
    batch_input_ids = torch.cat([self.paddingtensor2D(item['input_ids'], max_length) for item in features])
    batch_attention_mask = torch.cat([self.paddingtensor2D(item['attention_mask'], max_length) for item in features])
    batch_loss_mask = torch.cat([self.paddingtensor2D(item['loss_mask'], max_length) for item in features])

    batch = {'input_ids': batch_input_ids, 'attention_mask': batch_attention_mask, 'loss_mask': batch_loss_mask}
    return batch


def find_max_state_with_file(directory, filename='zero_to_fp32.py'):
  max_a = -1
  for subdir in os.listdir(directory):
    match = re.match(r'state_(\d+)', subdir)
    if match:
      a_value = int(match.group(1))
      subdir_path = os.path.join(directory, subdir)
      file_path = os.path.join(subdir_path, filename)
      if os.path.isdir(subdir_path) and os.path.exists(file_path):
        max_a = max(max_a, a_value)
  if max_a == -1:
    return None, 0
  return f'{directory}/state_{max_a}', max_a + 1


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='EAGLE-3 Training with Mixed Datasets')
  parser.add_argument('--basepath', type=str, default='meta-llama/Llama-3.1-8B-Instruct',
                     help='Path or model name for the base model (can be HuggingFace model name)')
  parser.add_argument('--cache_dir', type=str, default='./data_cache',
                     help='Directory to cache downloaded datasets')
  parser.add_argument('--trainpath', type=str, default=None,
                     help='Path to training data (will auto-generate if not provided)')
  parser.add_argument('--testpath', type=str, default=None,
                     help='Path to test data (will auto-generate if not provided)')
  parser.add_argument('--savedir', type=str, default='./checkpoints',
                     help='Directory to save model checkpoints')
  parser.add_argument('--local_rank', type=int, default=-1, help='local_rank for distributed training on gpus')
  parser = deepspeed.add_config_arguments(parser)
  args = parser.parse_args()

  # Download and mix datasets if paths not provided
  if args.trainpath is None or args.testpath is None:
    print("Downloading and mixing UltraChat_200k and shareGPT datasets...")
    train_path, test_path = download_and_mix_datasets(cache_dir=args.cache_dir)
    if args.trainpath is None:
      args.trainpath = train_path
    if args.testpath is None:
      args.testpath = test_path

  deepspeed_config = args.deepspeed_config
  with open(deepspeed_config) as f:
    ds_config = json.load(f)
  train_config = {
    'bs': ds_config['train_micro_batch_size_per_gpu'],
    'num_epochs': 40,
    'num_workers': 54,
    'max_len': 131072,
    'config_path': 'config.json',
  }

  tokenizer = AutoTokenizer.from_pretrained(args.basepath)
  traindataset = build_dataset_rank(tokenizer, args.trainpath)
  testdataset = build_dataset_rank(tokenizer, args.testpath)

  config = EConfig.from_pretrained(train_config['config_path'])
  model = Model(config, path=args.basepath, load_emb=True, load_head=True)
  model.scandata(args.trainpath, args.basepath)

  criterion = nn.SmoothL1Loss(reduction='none')

  num_epochs = train_config['num_epochs']

  model_engine, optimizer, _, _ = deepspeed.initialize(args=args, model=model, model_parameters=model.parameters())

  global_rank = deepspeed.comm.get_rank()
  rank = deepspeed.comm.get_local_rank()
  world_size = deepspeed.comm.get_world_size()
  if global_rank == 0:
    import wandb

    wandb.login(key='')
    wandb.init(project='eagle3-llama', entity='training', config=ds_config)

  os.makedirs(args.savedir, exist_ok=True)

  sampler = DistributedSampler(testdataset, num_replicas=world_size, rank=global_rank, shuffle=False)
  test_loader = DataLoader(
    testdataset,
    batch_size=train_config['bs'],
    sampler=sampler,
    num_workers=4,
    pin_memory=True,
    collate_fn=DataCollatorWithPadding(),
  )

  train_sampler = DistributedSampler(traindataset, num_replicas=world_size, rank=global_rank, shuffle=True)
  train_loader = DataLoader(
    traindataset,
    batch_size=train_config['bs'],
    sampler=train_sampler,
    num_workers=4,
    pin_memory=True,
    collate_fn=DataCollatorWithPadding(),
  )

  checkpoint_path, start_epoch = find_max_state_with_file(args.savedir)
  if checkpoint_path:
    print(f'load from {checkpoint_path}')
    model_engine.load_checkpoint(checkpoint_path)

  for epoch in range(start_epoch, num_epochs):
    train_sampler.set_epoch(epoch + 1)
    print(f'Now training epoch {epoch}')

    model.train()
    epoch_acces = [[] for _ in range(model.length)]
    epoch_plosses = [[] for _ in range(model.length)]

    for batch_idx, data in enumerate(tqdm(train_loader)):
      model.zero_grad()

      plosses, vlosses, acces = model_engine(
        input_ids=data['input_ids'].to(rank),
        attention_mask=data['attention_mask'].to(rank),
        loss_mask=data['loss_mask'],
      )

      ploss_weight = [0.8**i for i in range(len(plosses))]
      ploss = sum([ploss_weight[i] * plosses[i] for i in range(len(plosses))])
      loss = ploss
      model_engine.backward(loss)

      model_engine.step()

      if global_rank == 0:
        logdict = {'train/lr': optimizer.optimizer.param_groups[0]['lr']}
        for i in range(len(plosses)):
          logdict[f'train/ploss_{i}'] = plosses[i].item()
        for i in range(len(acces)):
          logdict[f'train/acc_{i}'] = acces[i]
        wandb.log(logdict)
      epoch_acces = [epoch_acces[i] + [acces[i]] for i in range(len(acces))]
      epoch_plosses = [epoch_plosses[i] + [plosses[i].item()] for i in range(len(plosses))]

    for i in range(len(epoch_acces)):
      acc_i = torch.tensor(epoch_acces[i]).cuda().mean()
      deepspeed.comm.all_reduce(acc_i, op=deepspeed.comm.ReduceOp.AVG)
      acc_i = acc_i.item()
      if global_rank == 0:
        wandb.log({f'train/epochacc_{i}': acc_i})
        print(f'Train Epoch [{epoch + 1}/{num_epochs}], position {i},  Acc: {acc_i:.2f}')

    for i in range(len(epoch_plosses)):
      loss_i = torch.tensor(epoch_plosses[i]).cuda().mean()
      deepspeed.comm.all_reduce(loss_i, op=deepspeed.comm.ReduceOp.AVG)
      loss_i = loss_i.item()
      if global_rank == 0:
        wandb.log({f'train/epochploss_{i}': loss_i})
        print(f'Train Epoch [{epoch + 1}/{num_epochs}], position {i}, pLoss: {loss_i:.2f}')

    epoch_acces = [[] for _ in range(model.length)]
    epoch_plosses = [[] for _ in range(model.length)]

    for batch_idx, data in enumerate(tqdm(test_loader)):
      with torch.no_grad():
        plosses, vlosses, acces = model_engine(
          input_ids=data['input_ids'].to(rank),
          attention_mask=data['attention_mask'].to(rank),
          loss_mask=data['loss_mask'],
        )
        epoch_acces = [epoch_acces[i] + [acces[i]] for i in range(len(acces))]
        epoch_plosses = [epoch_plosses[i] + [plosses[i].item()] for i in range(len(plosses))]

    for i in range(len(epoch_acces)):
      acc_i = torch.tensor(epoch_acces[i]).cuda().mean()
      deepspeed.comm.all_reduce(acc_i, op=deepspeed.comm.ReduceOp.AVG)
      acc_i = acc_i.item()
      if global_rank == 0:
        wandb.log({f'test/epochacc_{i}': acc_i})
        print(f'Test Epoch [{epoch + 1}/{num_epochs}], position {i},  Acc: {acc_i:.2f}')

    for i in range(len(epoch_plosses)):
      loss_i = torch.tensor(epoch_plosses[i]).cuda().mean()
      deepspeed.comm.all_reduce(loss_i, op=deepspeed.comm.ReduceOp.AVG)
      loss_i = loss_i.item()
      if global_rank == 0:
        wandb.log({f'test/epochploss_{i}': loss_i})
        print(f'Test Epoch [{epoch + 1}/{num_epochs}], position {i}, pLoss: {loss_i:.2f}')

    model_engine.save_16bit_model(f'{args.savedir}/state_{epoch}', exclude_frozen_parameters=True)
    if epoch % 10 == 0:
      deepspeed.DeepSpeedEngine.save_checkpoint(model_engine, save_dir=f'{args.savedir}/state_{epoch}')
