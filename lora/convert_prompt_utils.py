import re

re_attention = re.compile(r"""
\\\(|
\\\)|
\\\[|
\\]|
\\\\|
\\|
\(|
\[|
:([+-]?[.\d]+)\)|
\)|
]|
[^\\()\[\]:]+|
:
""", re.X)

re_break = re.compile(r"\s*\bBREAK\b\s*", re.S)

def replace_lora(line):
    # 使用正则表达式匹配出<lora:XXX:1.5>格式的字符串
    p = re.compile(r'<lora:([\w\d]+):([\d\.]+)>')
    m = p.findall(line)
    # 遍历匹配结果并替换
    for match in m:
        line = line.replace(f'<lora:{match[0]}:{match[1]}>', f'withLora({match[0]}, {match[1]})')
    return line

def find_and_replace_lora(line):
    # 使用正则表达式匹配出<lora:XXX:1.5>格式的字符串
    lora_list = []
    p = re.compile(r'<lora:([\w\d]+):([\d\.]+)>')
    m = p.findall(line)
    # 遍历匹配结果并替换
    for match in m:
        line = line.replace(f'<lora:{match[0]}:{match[1]}>', '')
        lora_list.append((match[0],float(match[1])))
    return line.strip(),lora_list


def extract_loraInfo(lora_info):
    return lora_info.model, lora_info.weight


def detect_lora(compelV1, prompt):
    lora_model_list = []
    prompt_info = compelV1.parse_prompt_string(prompt)
    lora_infos = prompt_info.lora_weights
    if len(lora_infos) < 1:
        return lora_model_list

    for lora_info in lora_infos:
        lora_id, weight = extract_loraInfo(lora_info)
        lora_model_list.append((lora_id, weight))
    return lora_model_list

def get_token_weight(text):

    res = []
    round_brackets = []
    square_brackets = []

    round_bracket_multiplier = 1.1
    square_bracket_multiplier = 1 / 1.1

    def multiply_range(start_position, multiplier):
        for p in range(start_position, len(res)):
            res[p][1] *= multiplier

    for m in re_attention.finditer(text):
        text = m.group(0)
        weight = m.group(1)

        if text.startswith('\\'):
            res.append([text[1:], 1.0])
        elif text == '(':
            round_brackets.append(len(res))
        elif text == '[':
            square_brackets.append(len(res))
        elif weight is not None and len(round_brackets) > 0:
            multiply_range(round_brackets.pop(), float(weight))
        elif text == ')' and len(round_brackets) > 0:
            multiply_range(round_brackets.pop(), round_bracket_multiplier)
        elif text == ']' and len(square_brackets) > 0:
            multiply_range(square_brackets.pop(), square_bracket_multiplier)
        else:
            parts = re.split(re_break, text)
            for i, part in enumerate(parts):
                if i > 0:
                    res.append(["BREAK", -1])
                res.append([part, 1.0])

    for pos in round_brackets:
        multiply_range(pos, round_bracket_multiplier)

    for pos in square_brackets:
        multiply_range(pos, square_bracket_multiplier)

    if len(res) == 0:
        res = [["", 1.0]]

    # merge runs of identical weights
    i = 0
    while i + 1 < len(res):
        if res[i][1] == res[i + 1][1]:
            res[i][0] += res[i + 1][0]
            res.pop(i + 1)
        else:
            i += 1

    return res

def convert_prompt(prompt):
    prompt_new = ''
    parse_prompt = get_token_weight(prompt)
    for temp in parse_prompt:
        text, weight = temp
        text = replace_lora(text)
        if weight > 1.0:
            weight = round(weight, 2)
            prompt_new = prompt_new + f"(({text}){weight})"
            continue
        prompt_new = prompt_new + text

    return prompt_new

def get_preEmbedding(model, prompt, negative_prompt,clip_skip=1):
    prompt_embeds = model(prompt, clip_skip)
    negativa_pormpts_embeds = model(negative_prompt, clip_skip)
    res = model.pad_prompt_tensor_same_length(prompt_emb=prompt_embeds, negative_prompt_emb=negativa_pormpts_embeds, CLIP_stop_at_last_layers=clip_skip)
    return res[0], res[1]