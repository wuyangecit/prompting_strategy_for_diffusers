import os
import torch
import safetensors


class Embedding:
    def __init__(self, vec, name, step=None):
        self.vec = vec
        self.name = name
        self.step = step
        self.shape = None
        self.vectors = 0
        self.cached_checksum = None
        self.sd_checkpoint = None
        self.sd_checkpoint_name = None
        self.optimizer_state_dict = None
        self.filename = None




class TextualInversionPlug:

    def __init__(self,embedding_dir,device="cpu",tokenizer=None, dtype=torch.float16):
        self.ids_lookup = {}
        self.embedding_dir = embedding_dir
        self.device = device
        self.dtype = dtype
        self.tokenizer = tokenizer

    def load_from_file(self,path, filename):
        name, ext = os.path.splitext(filename)
        ext = ext.upper()

        if ext in ['.BIN', '.PT']:
            data = torch.load(path, map_location="cpu")
        elif ext in ['.SAFETENSORS']:
            data = safetensors.torch.load_file(path, device=self.device)
        else:
            return

        # textual inversion embeddings
        if 'string_to_param' in data:
            param_dict = data['string_to_param']
            param_dict = getattr(param_dict, '_parameters',
                                 param_dict)
            assert len(param_dict) == 1, 'embedding file has multiple terms in it'
            emb = next(iter(param_dict.items()))[1]
        # diffuser concepts
        elif type(data) == dict and type(next(iter(data.values()))) == torch.Tensor:
            assert len(data.keys()) == 1, 'embedding file has multiple terms in it'

            emb = next(iter(data.values()))
            if len(emb.shape) == 1:
                emb = emb.unsqueeze(0)
        else:
            raise Exception(
                f"Couldn't identify {filename} as neither textual inversion embedding nor diffuser concept.")

        vec = emb.detach().to("cpu", dtype=self.dtype)
        embedding = Embedding(vec, name)
        embedding.step = data.get('step', None)
        embedding.sd_checkpoint = data.get('sd_checkpoint', None)
        embedding.sd_checkpoint_name = data.get('sd_checkpoint_name', None)
        embedding.vectors = vec.shape[0]
        embedding.shape = vec.shape[-1]
        embedding.filename = path
        self.register_embedding(embedding)

        return embedding

    def register_embedding(self, embedding):
        ids = self.tokenize_local([embedding.name])[0]

        first_id = ids[0]
        if first_id not in self.ids_lookup:
            self.ids_lookup[first_id] = []

        self.ids_lookup[first_id] = sorted(self.ids_lookup[first_id] + [(ids, embedding)], key=lambda x: len(x[0]),
                                           reverse=True)
        return embedding

    def load_textual_inversion(self):
        self.ids_lookup.clear()
        for file_name in os.listdir(self.embedding_dir):
            self.load_from_file(os.path.join(self.embedding_dir, file_name), file_name)

    def find_embedding_at_position(self, tokens, offset):
        token = tokens[offset]
        possible_matches = self.ids_lookup.get(token, None)

        if possible_matches is None:
            return None, None

        for ids, embedding in possible_matches:
            if tokens[offset:offset + len(ids)] == ids:
                return embedding, len(ids)

        return None, None

    def tokenize_local(self,texts):
        tokenized = self.tokenizer(texts, truncation=False, add_special_tokens=False)["input_ids"]

        return tokenized
