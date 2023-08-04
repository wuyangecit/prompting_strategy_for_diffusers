from torch import Tensor
from transformers import CLIPTokenizer, CLIPTextModel
from textual_inversion import TextualInversionPlug
from typing import Union, Optional, Callable, List, Tuple
from collections import namedtuple
from convert_prompt import *
from text_encoder_hijack import TextEncoderHijack
import torch
import open_clip.tokenizer

tokenizer_open_clip = open_clip.tokenizer._tokenizer

chunk_length = 75
id_end = tokenizer_open_clip.encoder["<end_of_text>"]
id_start = tokenizer_open_clip.encoder["<start_of_text>"]
id_pad = id_end
comma_token = [v for k, v in tokenizer_open_clip.encoder.items() if k == ',</w>'][0]
comma_padding_backtrack = 20
Max_prompt_cache_size = 500

class PromptChunk:
    def __init__(self):
        self.tokens = []
        self.multipliers = []
        self.fixes = []


PromptChunkFix = namedtuple('PromptChunkFix', ['offset', 'embedding'])

class EmbeddingExtent:
    def __init__(self,
                 tokenizer: CLIPTokenizer,
                 text_encoder: CLIPTextModel,
                 textual_inversion_manager: Optional[TextualInversionPlug] = None,
                 truncate_long_prompts: bool = True,
                 device: Optional[str] = None,
                 dtype = torch.float16,
                 hijack: Optional[TextEncoderHijack] = None,
                 ):
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        if textual_inversion_manager != None:
            self.textual_inversion_manager = textual_inversion_manager
        else:
            self.textual_inversion_manager = None
        self.truncate_long_prompts = truncate_long_prompts
        self.device = device
        self.dtype = dtype
        self.hijack = hijack
        self.CLIP_stop_at_last_layers = 1

        @property
        def device(self):
            return self._device if self._device else "cpu"

    @torch.no_grad()
    def __call__(self, texts: Union[str, List[str]], CLIP_stop_at_last_layers: int = 1) -> Tensor:
        if type(texts) != list:
            texts = [texts]
        batch_chunks, token_count = self.process_texts(texts)

        chunk_count = max([len(x) for x in batch_chunks])

        zs = []
        for i in range(chunk_count):
            batch_chunk = [chunks[i] if i < len(chunks) else self.empty_chunk() for chunks in batch_chunks]

            tokens = [x.tokens for x in batch_chunk]
            multipliers = [x.multipliers for x in batch_chunk]
            self.hijack.fixes = [x.fixes for x in batch_chunk]

            z = self.process_tokens(tokens, multipliers, CLIP_stop_at_last_layers)
            zs.append(z)

        return torch.hstack(zs)

    def tokenize_line(self, line):
        line = get_token_weight(line)

        tokenized = self.tokenize_local([text for text, _ in line])

        chunks = []
        chunk = PromptChunk()
        token_count = 0
        last_comma = -1

        def next_chunk(is_last=False):
            nonlocal token_count
            nonlocal last_comma
            nonlocal chunk

            if is_last:
                token_count += len(chunk.tokens)
            else:
                token_count += chunk_length

            to_add = chunk_length - len(chunk.tokens)
            if to_add > 0:
                chunk.tokens += [id_end] * to_add
                chunk.multipliers += [1.0] * to_add

            chunk.tokens = [id_start] + chunk.tokens + [id_end]
            chunk.multipliers = [1.0] + chunk.multipliers + [1.0]

            last_comma = -1
            chunks.append(chunk)
            chunk = PromptChunk()

        for tokens, (text, weight) in zip(tokenized, line):
            if text == 'BREAK' and weight == -1:
                next_chunk()
                continue

            position = 0
            while position < len(tokens):
                token = tokens[position]

                if token == comma_token:
                    last_comma = len(chunk.tokens)


                elif comma_padding_backtrack != 0 and len(chunk.tokens) == chunk_length and last_comma != -1 and len(
                        chunk.tokens) - last_comma <= comma_padding_backtrack:
                    break_location = last_comma + 1

                    reloc_tokens = chunk.tokens[break_location:]
                    reloc_mults = chunk.multipliers[break_location:]

                    chunk.tokens = chunk.tokens[:break_location]
                    chunk.multipliers = chunk.multipliers[:break_location]

                    next_chunk()
                    chunk.tokens = reloc_tokens
                    chunk.multipliers = reloc_mults

                if len(chunk.tokens) == chunk_length:
                    next_chunk()

                if self.textual_inversion_manager == None:
                    embedding = None
                else:
                    embedding, embedding_length_in_tokens = self.textual_inversion_manager.find_embedding_at_position(tokens, position)
                if embedding is None:
                    chunk.tokens.append(token)
                    chunk.multipliers.append(weight)
                    position += 1
                    continue

                emb_len = int(embedding.vec.shape[0])
                if len(chunk.tokens) + emb_len > chunk_length:
                    next_chunk()

                chunk.fixes.append(PromptChunkFix(len(chunk.tokens), embedding))

                chunk.tokens += [0] * emb_len
                chunk.multipliers += [weight] * emb_len
                position += embedding_length_in_tokens

        if len(chunk.tokens) > 0 or len(chunks) == 0:
            next_chunk(is_last=True)

        return chunks, token_count

    def process_texts(self, prompts):
        token_count = 0

        cache = {}
        batch_chunks = []
        for line in prompts:
            if line in cache:
                chunks = cache[line]
                if len(chunks) > Max_prompt_cache_size:
                    cache.clear()
            else:
                chunks, current_token_count = self.tokenize_line(line)
                token_count = max(current_token_count, token_count)
                cache[line] = chunks

            batch_chunks.append(chunks)

        return batch_chunks, token_count

    def process_tokens(self, remade_batch_tokens, batch_multipliers, CLIP_stop_at_last_layers):

        tokens = torch.asarray(remade_batch_tokens).to("cuda")

        z = self.encode_with_transformers(tokens, CLIP_stop_at_last_layers)
        # restoring original mean is likely not correct, but it seems to work well to prevent artifacts that happen otherwise
        batch_multipliers = torch.asarray(batch_multipliers).to("cuda")
        original_mean = z.mean().to(self.dtype)
        z = z * batch_multipliers.reshape(batch_multipliers.shape + (1,)).expand(z.shape)
        new_mean = z.mean().to(self.dtype)
        z = z * (original_mean / new_mean)
        return z


    def tokenize_local(self,texts):
        tokenized = self.tokenizer(texts, truncation=False, add_special_tokens=False)["input_ids"]

        return tokenized

    def empty_chunk(self):

        chunk = PromptChunk()
        chunk.tokens = [id_start] + [id_end] * (chunk_length + 1)
        chunk.multipliers = [1.0] * (chunk_length + 2)
        return chunk

    def encode_with_transformers(self, tokens, CLIP_stop_at_last_layers):
        outputs = self.text_encoder(input_ids=tokens, output_hidden_states=-CLIP_stop_at_last_layers)
        if CLIP_stop_at_last_layers > 1:
            z = outputs.hidden_states[-CLIP_stop_at_last_layers]
            z = self.text_encoder.text_model.final_layer_norm(z)
        else:
            z = outputs.last_hidden_state

        return z

    def pad_prompt_tensor_same_length(self, prompt_emb: torch.Tensor,
                                      negative_prompt_emb: torch.Tensor,
                                      CLIP_stop_at_last_layers: int = 1,
                                      ) -> List[torch.Tensor]:
        if len(prompt_emb.shape) != 3 or len(negative_prompt_emb.shape) != 3:
            raise ValueError("Prompt embeddings must be batched")
        res_list = []
        prompt_shape = prompt_emb[0].shape[0]
        negative_prompt_shape = negative_prompt_emb[0].shape[0]

        if prompt_shape == negative_prompt_shape:
            res_list.append(prompt_emb)
            res_list.append(negative_prompt_emb)
            return res_list
        elif prompt_shape > negative_prompt_shape:
            n = self.calc_diff(prompt_shape, negative_prompt_shape)
            if n == -1:
                raise ValueError("Prompt shape must be a multiple of 77")
            empty_tensor = [self.__call__([""], CLIP_stop_at_last_layers)] * n
            negative_prompt_emb = torch.hstack([negative_prompt_emb] + empty_tensor)
            res_list.append(prompt_emb)
            res_list.append(negative_prompt_emb)
            return res_list
        else:
            negative_prompt_emb = negative_prompt_emb[:, :prompt_shape, :]
            res_list.append(prompt_emb)
            res_list.append(negative_prompt_emb)
            return res_list

    def calc_diff(self, a, b):
        if a % 77 != 0 or b % 77 != 0:
            return -1
        return (a - b) // 77
