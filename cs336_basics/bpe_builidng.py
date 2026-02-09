import regex as re
import os
from concurrent.futures import ProcessPoolExecutor
from typing import BinaryIO

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def pre_tokenize_split(filepath, bound_st, bound_ed, pattern, special_tokens):
    with open(filepath, "rb") as f:
        f.seek(bound_st)
        chunk = f.read(bound_ed - bound_st).decode("utf-8", errors="ignore")
        special_pat = "|".join(map(re.escape, special_tokens))
        chunk_set = [s for s in re.split(special_pat, chunk) if s]
        corpus_weights = {}
        for small_chunk in chunk_set:
            splited_text = re.findall(pattern, small_chunk)
            for words in splited_text:
                data_u8 = words.encode("utf-8")
                corpus_weights[data_u8] = corpus_weights.get(data_u8, 0) + 1
    return corpus_weights

def build_seq_weights(filepath, num_process, special_tokens, PAT):
    with open(filepath, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_process, b"<|endoftext|>")
        
    parellel_params = [(filepath, start, end, PAT, special_tokens) for start, end in zip(boundaries[:-1], boundaries[1:])]
    
    with ProcessPoolExecutor(max_workers=num_process) as ex:
        results = list(ex.map(pre_tokenize_split, *zip(*parellel_params)))
        
    seq_weights = {} #{tuple: int(freq)}

    for dic in results:
        for k,v in dic.items():
            tuple_k = tuple(k)
            seq_weights[tuple_k] = seq_weights.get(tuple_k,0) + v
    
    return seq_weights

def build_pair_cnt(_seq_weights):
    _pair_cnt = {}
    for k,v in _seq_weights.items():
        for ch1,ch2 in zip(k[:-1],k[1:]):
            pair = (ch1,ch2)
            _pair_cnt[pair] = _pair_cnt.get(pair,0) + v
    return _pair_cnt

def find_max(_pair_cnt,_token_dict):
    maxcnt = -1
    maxpair = None
    bytes_pair = None
    for p,v in _pair_cnt.items():
        if v == maxcnt:
            bytes_pair_new = _token_dict[p[0]],_token_dict[p[1]]
            maxpair = p if bytes_pair_new>bytes_pair else maxpair
        if v > maxcnt:
            maxcnt = v
            maxpair = p
            bytes_pair = _token_dict[p[0]],_token_dict[p[1]]
    return maxpair,maxcnt

def merge_operation(_seq_weights, merge_pair, merge_id):
    _seq_weights_copy = {}
    for k, v in _seq_weights.items():
        _ = 0
        new_k = []
        while _ < len(k):
            if _ + 1 < len(k) and (k[_],k[_+1])==merge_pair:
                new_k.append(merge_id)
                _ += 2
            else:
                new_k.append(k[_])
                _ += 1
        new_k = tuple(new_k)
        _seq_weights_copy[new_k] = _seq_weights_copy.get(new_k,0)+v
    return _seq_weights_copy

def my_train_bpe(filepath,vocab_size,special_tokens,PAT,num_processes=4):
    token_dict = {i:bytes([i]) for i in range(256)}
    merge_list = []
    seq_now = build_seq_weights(filepath,num_processes,special_tokens,PAT)
    merge_num = vocab_size - len(special_tokens) - 256
    token_id = 256
    for i in range(merge_num):
        pairnow,freq = find_max(build_pair_cnt(seq_now),token_dict)
        seq_now = merge_operation(seq_now,pairnow,token_id)
        token_dict[token_id] = token_dict[pairnow[0]]+token_dict[pairnow[1]] 
        merge_list.append((token_dict[pairnow[0]],token_dict[pairnow[1]]))
        token_id += 1
    for s in special_tokens:
        token_dict[token_id] = bytes(s)
        token_id += 1
    return token_dict, merge_list