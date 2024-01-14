import streamlit as st
import pandas as pd
import pickle
from datetime import datetime

import lightGE.core.nn
# Store the initial value of widgets in session state
from lightGE.compoment.model_creator import ModelCreator
from lightGE.compoment.controller import Controller
from lightGE.core.nn import *

# -- Set page config
pagetitle = 'nnengine-create'

st.set_page_config(page_title=pagetitle, page_icon=":eyeglasses:")

st.title("模型创建")

if "data" not in st.session_state:
    st.session_state.data = [

    ]

if 'model_save_path' not in st.session_state:
    st.session_state.model_save_path = ''

if 'model_repr' not in st.session_state:
    st.session_state.model_repr = ''

if 'model_create_time' not in st.session_state:
    st.session_state.model_create_time = None

if 'model' not in st.session_state:
    st.session_state.model = None

if 'model_uploaded' not in st.session_state:
    st.session_state.model_uploaded = False
    
if 'controller' not in st.session_state:
    st.session_state.controller = Controller()

name_dict = {
    "Linear": {
        "type": "模型类型",
        "n_inputs": "输入维度",
        "n_outputs": "输出维度",
    },
    "Conv2d": {
        "type": "模型类型",
        "n_inputs": "输入维度",
        "n_outputs": "输出维度",
        "filter_size": "卷积核大小",
        "stride": "步长",
        "padding": "填充",
        "bias": "是否加上偏置矩阵"
    },
    "MaxPool2d": {
        "type": "模型类型",
        "filter_size": "池化核大小",
        "stride": "步长",
        "padding": "填充",

    },
    "AvgPool2d": {
        "type": "模型类型",
        "filter_size": "池化核大小",
        "stride": "步长",
        "padding": "填充",

    },
    "LSTM": {
        "type": "模型类型",
        "n_inputs": "输入维度",
        "n_outputs": "输出维度",

    },
    "Tanh": {
        "type": "模型类型"
    },
    "Sigmoid": {
        "type": "模型类型"
    },
    "ReLu": {
        "type": "模型类型"
    },
    'Softmax': {
        'type': '模型类型'
    },
    "BatchNorm1d": {
        "type": "模型类型",
        "n_inputs": "输入维度"
    },
    "BatchNorm2d": {
        "type": "模型类型",
        "n_inputs": "输入维度"
    },
    "Dropout": {
        "type": "模型类型",
        "p": "保留率"
    },
    "Dropout2d": {
        "type": "模型类型",
        "p": "保留率"
    },
    "MNIST": {
        "type": "模型类型",
    },
    "SelfAttention": {
        "type": "模型类型",
        "n_inputs": "输入维度",
        "n_outputs": "输出维度",
        "mask": "是否使用mask"
    },
    "MultiAttention": {
        "type": "模型类型",
        "n_heads": "自注意力头的个数",
        "n_inputs": "输入维度",
        "mask": "是否使用mask"
    },
    "AddNorm": {
        "type": "模型类型",
        "sentence_len": "句子长度",
        "n_inputs": "输入维度"
    },
    "FeedForward": {
        "type": "模型类型",
        "n_inputs": "输入维度",
        "ndrop_prob": "保留率"
    },
    "PositionEmbedding": {
        "type": "模型类型",
        "embedding_len": "嵌入长度",
        "sentence_len": "句子长度"
    },
    "EncoderBlock": {
        "type": "模型类型",
        "n_inputs": "输入维度",
        "sentence_len": "句子长度",
        "n_heads": "多头自注意力头数",
        "batch": "批处理大小",
        "hidden_feedforward": "feedforward中间隐藏层的个数",
        "ndrop_prob": "保留率"
    },
    "Encoder": {
        "type": "模型类型",
        "n_inputs": "输入维度",
        "sentence_len": "句子长度",
        "n_heads": "多头自注意力头数",
        "batch": "批处理大小",
        "hidden_feedforward": "feedforward中间隐藏层的个数",
        "ndrop_prob": "保留率",
        "block_num": "block的个数"
    },
    "DecoderBlock": {
        "type": "模型类型",
        "n_inputs": "输入维度",
        "sentence_len": "句子长度",
        "n_heads": "多头自注意力头数",
        "batch": "批处理大小",
        "hidden_feedforward": "feedforward中间隐藏层的个数",
        "ndrop_prob": "保留率"
    },
    "Decoder": {
        "type": "模型类型",
        "n_inputs": "输入维度",
        "sentence_len": "句子长度",
        "n_heads": "多头自注意力头数",
        "batch": "批处理大小",
        "hidden_feedforward": "feedforward中间隐藏层的个数",
        "ndrop_prob": "保留率",
        "block_num": "block的个数"
    },
    "Transformer": {
        "type": "模型类型",
        "encoder_block": "encoder块的个数",
        "decoder_block": "decoder块的个数",
        "batch": "批处理大小",

        "sentence_len": "句子长度",
        "n_inputs": "输入维度",
        "n_heads": "多头自注意力头数",

        "hidden_feedforward": "feedforward中间隐藏层的个数",
        "vocab_len": "单词表的大小",
        "ndrop_prob": "保留率"
    },
}

init_val = {
    "Conv2d": {
        "stride": 1,
        "padding": 0,
        "bias": 1
    },
    "MaxPool2d": {
        "stride": 1,
        "padding": 0,
    },
    "AvgPool2d": {
        "stride": 1,
        "padding": 0,
    },
    "Dropout": {
        "p": 0.5
    },
    "Dropout2d": {
        "p": 0.5
    },
    "SelfAttention": {
        "mask": 0
    },
    "MultiAttention": {
        "mask": 0
    },
    "FeedForward": {
        "ndrop_prob": 0.9
    },
    'Encoder': {
        'ndrop_prob': 0.9
    },
    'EncoderBlock': {
        'ndrop_prob': 0.9
    },
    'Decoder': {
        'ndrop_prob': 0.9
    },
    'DecoderBlock': {
        'ndrop_prob': 0.9
    },
    "Transformer": {
        "encoder_block": 3,
        "decoder_block": 3,
        "batch": 8,

        "sentence_len": 5,
        "n_inputs": 512,
        "n_heads": 2,

        "hidden_feedforward": 2048,
        "vocab_len": 1000,
        'ndrop_prob': 0.9
    }
}

is_bool = {
    "Conv2d": ["bias"],
    "SelfAttention": ["mask"],
    "MultiAttention": ["mask"],
}

zero_positive_int = {
    'Conv2d': ['padding'],
    'MaxPool2d': ['padding']
}

zero_to_one_float = {
    'Dropout': ['p'],
    'Dropout2d': ['p'],
    'FeedForward': ['ndrop_prob'],
    'EncoderBlock': ['ndrop_prob'],
    'Encoder': ['ndrop_prob'],
    'DecoderBlock': ['ndrop_prob'],
    'Decoder': ['ndrop_prob'],
    'Transformer': ['ndrop_prob']
}


def plus_one():
    flag = 0
    for key in st.session_state.tmp:
        if st.session_state.tmp[key] is None:
            flag = 1
    if flag:
        st.warning('请配置所有必要参数', icon="⚠️")
    else:
        st.session_state.data.append(st.session_state.tmp.copy())
        st.session_state.tmp = {}
    return


def create_model(model_name):
    # with open("./config.json", "w") as file:
    #     json.dump(st.session_state.data, file, ensure_ascii=False,indent='\t')
    model = ModelCreator.create(st.session_state.data)
    if model is not None:
        st.session_state.model = model
        st.session_state.controller.config_model(st.session_state.model)
        st.session_state.model_repr = st.session_state.model.__repr__()
        st.session_state.model_create_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def upload_model():
    st.session_state.model_uploaded = True


def _delete(i):
    st.session_state.data.pop(i)


def set_param(key, value):
    st.session_state.tmp[key] = value


with st.sidebar:
    types = []
    for model_type in name_dict:
        types.append(model_type)
    model_name = st.text_input("模型名：", value="create")
    selected_type = st.selectbox(
        '选择要添加的模型类型',
        types)
    if selected_type:
        _tmp = {}
        for parameter_name in name_dict[selected_type]:
            if selected_type in init_val and parameter_name in init_val[selected_type]:
                _tmp[parameter_name] = init_val[selected_type][parameter_name]
            elif parameter_name == "type":
                _tmp[parameter_name] = selected_type
            else:
                _tmp[parameter_name] = 0
        st.session_state.tmp = _tmp.copy()
        with st.form("Model"):
            for parameter_name in name_dict[selected_type]:
                if parameter_name == "type":
                    continue
                if selected_type in init_val and parameter_name in init_val[selected_type]:
                    initial_value = init_val[selected_type][parameter_name]
                else:
                    initial_value = None
                if selected_type in is_bool and \
                        parameter_name in is_bool[selected_type]: # boolean类型输入框
                    st.session_state.tmp[parameter_name] = st.checkbox(
                        name_dict[selected_type][parameter_name],
                        key=parameter_name,
                        value=initial_value
                    )
                elif selected_type in zero_positive_int and \
                    parameter_name in zero_positive_int[selected_type]: # 最小值为0的整数类型输入框
                    st.session_state.tmp[parameter_name] = st.number_input(
                        name_dict[selected_type][parameter_name],
                        key=parameter_name,
                        value=0,
                        min_value=0,
                        placeholder='[请输入参数...]'
                    )
                elif selected_type in zero_to_one_float and \
                    parameter_name in zero_to_one_float[selected_type]: # 取值为0到1之间的浮点数类型输入框
                    st.session_state.tmp[parameter_name] = st.number_input(
                        name_dict[selected_type][parameter_name],
                        key=parameter_name,
                        value=0.5,
                        min_value=0.0,
                        max_value=1.0,
                        placeholder='[请输入参数...]'
                    )
                else: # 最小值为1的整数类型输入框
                    st.session_state.tmp[parameter_name] = st.number_input(
                        name_dict[selected_type][parameter_name],
                        key=parameter_name,
                        value=5,
                        min_value=1,
                        placeholder='[请输入参数...]'
                    )
            submitted = st.form_submit_button("确认")
            if submitted:
                plus_one()

named_data = []
for item in st.session_state.data:
    # model_data 用于记录单个模型的参数，如：
    # model_data = [['Linear'], ['n_inputs', '输入维度', '10'], ['n_outputs', '输出维度', '5']]
    # 列表的第一个元素为仅包含 模型名称 这一个元素的列表
    # 列表后续元素为包含 参数名称、参数描述、参数值 三个元素的列表
    model_data = [[item['type']]]
    for key in item:
        if key != 'type':
            row = [key, name_dict[item['type']][key], str(item[key])]
            model_data.append(row)
    named_data.append(model_data)

for i in range(len(st.session_state.data)):
    model_data = named_data[i]
    df = pd.DataFrame(data=model_data[1:], columns=['参数名称', '参数描述', '参数值'])
    expander = st.expander(f"model {i}: {model_data[0][0]}")  # model_data[0][0] 就是 模型名称
    expander.dataframe(data=df, use_container_width=True)
    expander.button("删除", on_click=_delete, args=[i], key="delete" + str(i))

st.button("创建模型", type="primary", on_click=create_model, args=(model_name, ))

uploaded_file = st.file_uploader('上传已有模型', on_change=upload_model)
if st.session_state.model_uploaded and uploaded_file is not None:
    with st.status('模型处理中...'):
        model = pickle.loads(uploaded_file.read())
        if isinstance(model, Model):
            st.session_state.model = model
            st.session_state.controller.config_model(model)
            st.session_state.model_repr = model.__repr__()
            st.session_state.model_create_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.write('模型处理成功✅')
        else:
            st.write('模型处理失败⚠️请保证所选择的模型为正确的类型')
    st.session_state.model_uploaded = False

if st.session_state.model_repr != '':
    with st.container(border=True):
        st.success(f'{st.session_state.model_create_time} 模型创建成功！', icon="✅")
        st.write("当前模型如下：")
        st.code(st.session_state.model_repr)
        print(id(st.session_state.model.__class__))
        print(id(lightGE.core.nn.LinearEndSequential))
        st.download_button('下载模型', data=pickle.dumps(st.session_state.model), file_name=f"{model_name}.model")
