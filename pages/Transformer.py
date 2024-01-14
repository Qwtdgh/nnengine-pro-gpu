import os
import pickle
from io import StringIO

import streamlit as st

from lightGE.core.transformer.transformer import Transformer, predict
from lightGE.compoment.controller import Controller
from lightGE.core.nn import *

MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB
check_eval = False
uploaded_file = None
global optimizer_adam
dataDir = ['res/simple_translation/simple_translation_en.txt', 'res/simple_translation/val_simple_translation_en.txt',
           'res/simple_translation/simple_translation_zh.txt', 'res/simple_translation/val_simple_translation_zh.txt']

# -- Set page config
pagetitle = 'nnengine-Transformer'

if 'hyper_param_dict' not in st.session_state:
    st.session_state.hyper_param_dict = dict()

if 'model_repr' not in st.session_state:
    st.session_state.model_repr = ''

if 'model_create_time' not in st.session_state:
    st.session_state.model_create_time = None

if 'model' not in st.session_state:
    st.session_state.model = None

if 'controller' not in st.session_state:
    st.session_state.controller = Controller()

# 文件上传状态变量
if 'transformer_training_src_uploaded' not in st.session_state:
    st.session_state.transformer_training_src_uploaded = False

if 'transformer_training_tgt_uploaded' not in st.session_state:
    st.session_state.transformer_training_tgt_uploaded = False

if 'transformer_eval_src_uploaded' not in st.session_state:
    st.session_state.transformer_eval_src_uploaded = False

if 'transformer_eval_tgt_uploaded' not in st.session_state:
    st.session_state.transformer_eval_tgt_uploaded = False

# 文件路径记录变量
if 'transformer_training_src_path' not in st.session_state:
    st.session_state.transformer_training_src_path = ''

if 'transformer_training_tgt_path' not in st.session_state:
    st.session_state.transformer_training_tgt_path = ''

if 'transformer_eval_src_path' not in st.session_state:
    st.session_state.transformer_eval_src_path = ''

if 'transformer_eval_tgt_path' not in st.session_state:
    st.session_state.transformer_eval_tgt_path = ''

def training_src_uploaded():
    st.session_state.transformer_training_src_uploaded = True

def training_tgt_uploaded():
    st.session_state.transformer_training_tgt_uploaded = True

def eval_src_uploaded():
    st.session_state.transformer_eval_src_uploaded = True

def eval_tgt_uploaded():
    st.session_state.transformer_eval_tgt_uploaded = True

st.set_page_config(page_title=pagetitle, page_icon=":eyeglasses:")

with st.sidebar:
    st.subheader('当前超参数配置', divider='blue')
    with st.container(border=True):
        st.json(st.session_state.hyper_param_dict)

st.title("Transformer")
train_tab, prediction_tab = st.tabs(['机器翻译任务-模型训练', '机器翻译任务-模型预测'])

with train_tab:
    with st.container(border=True):
        st.write('**机器翻译数据集选择**')
        # training src
        uploaded_training_src_file = st.file_uploader('选择训练集-源语言文本文件', type='.txt',
                                                      on_change=training_src_uploaded)
        if st.session_state.transformer_training_src_uploaded and uploaded_training_src_file is not None:
            with st.status('txt文件处理中...'):
                st.write('读取txt文件中...')
                file_path = 'res/translation/training_src.txt'
                if os.path.exists(file_path):
                    os.remove(file_path)
                stringio = StringIO(uploaded_training_src_file.getvalue().decode("utf-8"))
                content = stringio.read()
                content = content.replace('\r\n', '\n')
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                st.session_state.transformer_training_src_path = file_path
                st.write('txt文件处理完成✅')
            st.session_state.transformer_training_src_uploaded = False

        # training tgt
        uploaded_training_tgt_file = st.file_uploader('选择训练集-目标语言文本文件', type='.txt',
                                                      on_change=training_tgt_uploaded)
        if st.session_state.transformer_training_tgt_uploaded and uploaded_training_tgt_file is not None:
            with st.status('txt文件处理中...'):
                st.write('读取txt文件中...')
                file_path = 'res/translation/training_tgt.txt'
                if os.path.exists(file_path):
                    os.remove(file_path)
                stringio = StringIO(uploaded_training_tgt_file.getvalue().decode("utf-8"))
                content = stringio.read()
                content = content.replace('\r\n', '\n')
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                st.session_state.transformer_training_tgt_path = file_path
                st.write('txt文件处理完成✅')
            st.session_state.transformer_training_tgt_uploaded = False

        # eval src
        uploaded_eval_src_file = st.file_uploader('选择测试集-源语言文本文件', type='.txt',
                                                      on_change=eval_src_uploaded)
        if st.session_state.transformer_eval_src_uploaded and uploaded_eval_src_file is not None:
            with st.status('txt文件处理中...'):
                st.write('读取txt文件中...')
                file_path = 'res/translation/eval_src.txt'
                if os.path.exists(file_path):
                    os.remove(file_path)
                stringio = StringIO(uploaded_eval_src_file.getvalue().decode("utf-8"))
                content = stringio.read()
                content = content.replace('\r\n', '\n')
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                st.session_state.transformer_eval_src_path = file_path
                st.write('txt文件处理完成✅')
            st.session_state.classification_eval_set_uploaded = False

        # eval tgt
        uploaded_eval_tgt_file = st.file_uploader('选择测试集-目标语言文本文件', type='.txt',
                                                      on_change=eval_tgt_uploaded)
        if st.session_state.transformer_eval_tgt_uploaded and uploaded_eval_tgt_file is not None:
            with st.status('txt文件处理中...'):
                st.write('读取txt文件中...')
                file_path = 'res/translation/eval_tgt.txt'
                if os.path.exists(file_path):
                    os.remove(file_path)
                stringio = StringIO(uploaded_eval_tgt_file.getvalue().decode("utf-8"))
                content = stringio.read()
                content = content.replace('\r\n', '\n')
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                st.session_state.transformer_eval_tgt_path = file_path
                st.write('txt文件处理完成✅')
            st.session_state.classification_eval_tgt_uploaded = False

        if st.session_state.transformer_training_src_path != '' and \
            st.session_state.transformer_training_tgt_path != '' and \
            st.session_state.transformer_eval_src_path != '' and \
            st.session_state.transformer_eval_tgt_path != '':
            with st.status('数据集处理中...'):
                st.write('数据集处理中...')
                st.session_state.controller.config_data_path('Transformer',
                                                             [st.session_state.transformer_training_src_path,
                                                              st.session_state.transformer_eval_src_path,
                                                              st.session_state.transformer_training_tgt_path,
                                                              st.session_state.transformer_eval_tgt_path])
                st.write('数据集处理完成✅')
            st.write(f'最大句子长度为：{st.session_state.controller.sentence_loader.max_sentence_len}\n'
                     f'单词数量：{len(st.session_state.controller.sentence_loader.tgt_word2vec.wv.key_to_index)}\n')

        with st.expander('文本数据集展示'):
            st.write('训练集-源语言文本：')
            if st.session_state.transformer_training_src_path != '':
                count = 3
                content = ''
                for line in open(st.session_state.transformer_training_src_path, 'r', encoding='utf-8'):
                    if count > 0:
                        content += f'{line}'
                        count -= 1
                content += '......'
                st.code(content)
            else:
                st.write('> [请上传训练集-源语言文本]')

            st.write('训练集-目标语言文本：')
            if st.session_state.transformer_training_tgt_path != '':
                count = 3
                content = ''
                for line in open(st.session_state.transformer_training_tgt_path, 'r', encoding='utf-8'):
                    if count > 0:
                        content += f'{line}'
                        count -= 1
                content += '......'
                st.code(content)
            else:
                st.write('> [请上传训练集-目标语言文本]')

            st.write('测试集-源语言文本：')
            if st.session_state.transformer_eval_src_path != '':
                count = 3
                content = ''
                for line in open(st.session_state.transformer_eval_src_path, 'r', encoding='utf-8'):
                    if count > 0:
                        content += f'{line}'
                        count -= 1
                content += '......'
                st.code(content)
            else:
                st.write('> [请上传测试集-源语言文本]')

            st.write('测试集-目标语言文本：')
            if st.session_state.transformer_eval_tgt_path != '':
                count = 3
                content = ''
                for line in open(st.session_state.transformer_eval_tgt_path, 'r', encoding='utf-8'):
                    if count > 0:
                        content += f'{line}'
                        count -= 1
                content += '......'
                st.code(content)
            else:
                st.write('> [请上传测试集-目标语言文本]')

    train = st.button('开始训练！', type='primary')
    if train:
        with st.status('训练前准备...'):
            st.write('准备训练器')
            st.session_state.controller.prepare_trainer()
            st.write('准备训练器完成✅')
        if st.session_state.controller.train():
            st.write('训练完成！✅')
            st.download_button('下载模型', data=pickle.dumps(st.session_state.model), file_name=f"trained_model.model")

        # optimizer = Adam(parameters=model.parameters(), beta=(0.9, 0.98), d_model=word_vector_size,
                         # warmup_step=400)

with prediction_tab:
    input_text = st.text_input('**输入要预测的语句**', placeholder="Type text...")
    predict_begin = st.button(':sparkles: :rainbow[开始预测]', type='primary')
    if predict_begin:
        if st.session_state.controller.sentence_loader is None:
            st.error('请先配置数据集！', icon="🚨")
        model = st.session_state.controller.model_config.get_model()
        if len(model.sub_models)  == 1 and type(model.sub_models['0']).__name__ == 'Transformer':
            with st.status('正在预测...'):
                out = predict(input_text, st.session_state.controller.sentence_loader, model.sub_models['0'])
                out = out.replace('[END]', '')
                st.write('预测完成✅')
            st.subheader("输出结果")
            st.markdown(f'**{out}**')
        else:
            st.error('已选模型不是Transformer类型，请重新配置模型', icon="🚨")
