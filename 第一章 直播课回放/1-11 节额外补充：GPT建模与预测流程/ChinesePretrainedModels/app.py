import streamlit as st
from transformers import GPT2LMHeadModel, CpmTokenizer
import argparse
import os
import torch
import time
from utils import top_k_top_p_filtering
import torch.nn.functional as F
#pip install streamlit后测试streamlit hello
st.set_page_config(page_title="Demo", initial_sidebar_state="auto", layout="wide")


@st.cache(allow_output_mutation=True)
def get_model(device, model_path):
    tokenizer = CpmTokenizer(vocab_file="vocab/chinese_vocab.model")
    eod_id = tokenizer.convert_tokens_to_ids("<eod>")  # 文档结束符
    sep_id = tokenizer.sep_token_id
    unk_id = tokenizer.unk_token_id

    model = GPT2LMHeadModel.from_pretrained(model_path)
    model.to(device)
    model.eval()
    return tokenizer, model, eod_id, sep_id, unk_id



device_ids = 0
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICE"] = str(device_ids)
device = torch.device("cuda" if torch.cuda.is_available() and int(device_ids) >= 0 else "cpu")
tokenizer, model, eod_id, sep_id, unk_id = get_model(device, "model/novel/epoch50")

def generate_next_token(input_ids,args):
    """
    对于给定的上文，生成下一个单词
    """
    # 只根据当前位置的前context_len个token进行生成
    input_ids = input_ids[:, -200:]
    outputs = model(input_ids=input_ids)
    logits = outputs.logits
    # next_token_logits表示最后一个token的hidden_state对应的prediction_scores,也就是模型要预测的下一个token的概率
    next_token_logits = logits[0, -1, :]
    next_token_logits = next_token_logits / args.temperature
    # 对于<unk>的概率设为无穷小，也就是说模型的预测结果不可能是[UNK]这个token
    next_token_logits[unk_id] = -float('Inf')
    filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=args.top_k, top_p=args.top_p)
    # torch.multinomial表示从候选集合中选出无放回地进行抽取num_samples个元素，权重越高，抽到的几率越高，返回元素的下标
    next_token_id = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
    return next_token_id


def predict_one_sample(model, tokenizer, device, args, title, context):
    title_ids = tokenizer.encode(title, add_special_tokens=False)
    context_ids = tokenizer.encode(context, add_special_tokens=False)
    input_ids = title_ids + [sep_id] + context_ids
    cur_len = len(input_ids)
    last_token_id = input_ids[-1]  # 已生成的内容的最后一个token
    input_ids = torch.tensor([input_ids], dtype=torch.long, device=device)

    while True:
        next_token_id = generate_next_token(input_ids,args)
        input_ids = torch.cat((input_ids, next_token_id.unsqueeze(0)), dim=1)
        cur_len += 1
        word = tokenizer.convert_ids_to_tokens(next_token_id.item())
        # 超过最大长度，并且换行
        if cur_len >= args.generate_max_len and last_token_id == 8 and next_token_id == 3:
            break
        # 超过最大长度，并且生成标点符号
        if cur_len >= args.generate_max_len and word in [".", "。", "！", "!", "?", "？", ",", "，"]:
            break
        # 生成结束符
        if next_token_id == eod_id:
            break
    result = tokenizer.decode(input_ids.squeeze(0))
    content = result.split("<sep>")[1]  # 生成的最终内容
    return content


def writer():
    st.markdown(
        """
        ## GPT生成模型
        """
    )
    st.sidebar.subheader("配置参数")

    generate_max_len = st.sidebar.number_input("generate_max_len", min_value=0, max_value=512, value=32, step=1)
    top_k = st.sidebar.slider("top_k", min_value=0, max_value=10, value=3, step=1)
    top_p = st.sidebar.number_input("top_p", min_value=0.0, max_value=1.0, value=0.95, step=0.01)
    temperature = st.sidebar.number_input("temperature", min_value=0.0, max_value=100.0, value=1.0, step=0.1)

    parser = argparse.ArgumentParser()
    parser.add_argument('--generate_max_len', default=generate_max_len, type=int, help='生成标题的最大长度')
    parser.add_argument('--top_k', default=top_k, type=float, help='解码时保留概率最高的多少个标记')
    parser.add_argument('--top_p', default=top_p, type=float, help='解码时保留概率累加大于多少的标记')
    parser.add_argument('--max_len', type=int, default=512, help='输入模型的最大长度，要比config中n_ctx小')
    parser.add_argument('--temperature', type=float, default=temperature, help='输入模型的最大长度，要比config中n_ctx小')
    args = parser.parse_args()

    context = st.text_area("请输入标题", max_chars=512)
    title = st.text_area("请输入正文", max_chars=512)
    if st.button("点我生成结果"):
        start_message = st.empty()
        start_message.write("自毁程序启动中请稍等 10.9.8.7 ...")
        start_time = time.time()
        result = predict_one_sample(model, tokenizer, device, args, title, context)
        end_time = time.time()
        start_message.write("生成完成，耗时{}s".format(end_time - start_time))
        st.text_area("生成结果", value=result, key=None)
    else:
        st.stop()


if __name__ == '__main__':
    writer()
