import pickle

import numpy as np
import streamlit as st
from PIL import Image

from lightGE.compoment.controller import Controller
from lightGE.core import Tensor, MNIST
from lightGE.core.nn import *

from datetime import datetime

MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB
check_predict = False
uploaded_file = None

# -- Set page config
pagetitle = 'nnengine-Mnist'

if 'model_repr' not in st.session_state:
    st.session_state.model_repr = ''

if 'model_create_time' not in st.session_state:
    st.session_state.model_create_time = None

if 'model' not in st.session_state:
    st.session_state.model = None

if 'controller' not in st.session_state:
    st.session_state.controller = Controller()

if 'mnist_image_uploaded' not in st.session_state:
    st.session_state.mnist_image_uploaded = False

if 'hyper_param_dict' not in st.session_state:
    st.session_state.hyper_param_dict = dict()
st.set_page_config(page_title=pagetitle, page_icon=":eyeglasses:")


def image_uploaded():
    st.session_state.mnist_image_uploaded = True


def choose_mnist_model():
    st.session_state.model = MNIST()
    st.session_state.controller.config_model(st.session_state.model)
    st.session_state.model_repr = st.session_state.model.__repr__()
    st.session_state.model_create_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def choose_mnist_trained_model():
    st.session_state.model = pickle.load(open('tmp/mnist.pkl', 'rb'))
    st.session_state.controller.config_model(st.session_state.model)
    st.session_state.model_repr = st.session_state.model.__repr__()
    st.session_state.model_create_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

with st.sidebar:
    st.subheader('当前超参数配置', divider='blue')
    with st.container(border=True):
        st.json(st.session_state.hyper_param_dict)

st.title("Mnist")
st.session_state.controller.config_data_path('MNIST', data_paths=['./res/mnist/'])
train_tab, prediction_tab = st.tabs(['手写数字识别任务-模型训练', '手写数字识别任务-模型预测'])
with train_tab:
    st.subheader('当前选择模型', divider='blue')
    with st.expander("模型结构"):
        st.code(st.session_state.model_repr)
    st.button('选择 MNIST Demo 新模型', on_click=choose_mnist_model)
    st.button('选择 MNIST Demo 已训练好的模型', on_click=choose_mnist_model)
    train = st.button('开始训练！', type='primary')
    if train:
        with st.status('训练前准备...'):
            st.write('准备训练器')
            st.session_state.controller.prepare_trainer()
            st.write('准备训练器完成✅')
        if st.session_state.controller.train():
            st.write('训练完成！✅')
            st.download_button('下载模型', data=pickle.dumps(st.session_state.model), file_name=f"trained_model.model")

with prediction_tab:
    image = None
    y_pred = None
    uploaded_image_file = st.file_uploader('选择待预测的图片', type=["png", "jpg", "jpeg"], on_change=image_uploaded)
    if st.session_state.mnist_image_uploaded and uploaded_image_file is not None:
        with st.status('图片处理中...'):
            st.write('图片处理中...')

            # 读取图片并转换为模型可以处理的格式
            image = Image.open(uploaded_image_file).convert('L')  # 转换为灰度图像
            image = image.resize((28, 28))  # 调整大小
            img_array = np.array(image) / 255.0  # 归一化
            # 将图片转换为模型需要的格式
            img_tensor = Tensor(img_array.reshape(1, 1, 28, 28))  # 添加额外的批处理和通道维度
            y_pred = st.session_state.controller.predict(img_tensor)
            y_pred = np.argmax(y_pred.data, axis=1)[0]
        st.session_state.mnist_image_uploaded = False

    col_pic, col_res = st.columns(2)
    with col_pic:
        con1 = st.container(border=True)
        with con1:
            st.header("图片内容")
            # 显示上传的图片
            if image is None:
                st.write('请上传图片')
            else:
                st.image(image, caption="上传的图片", use_column_width=True)
    with col_res:
        # 显示预测结果
        con2 = st.container(border=True)
        with con2:
            st.header("预测结果")
            st.markdown('**预测数字: {}**'.format(y_pred))

        # # 显示前3个图像和它们的预测
        # for i in range(3):
        #     x, y = eval_dataset[i]
        #     # 确保输入是四维的
        #     if len(x.shape) == 3:
        #         x = np.expand_dims(x, axis=0)
        #     y_pred = trainer.m(Tensor(x))
        #     y_pred = np.argmax(y_pred.data, axis=1)[0]
        #
        #     # 使用 Streamlit 的 st.image 显示图像和预测
        #     st.image(x.reshape(28, 28),use_column_width=True)
        #     st.markdown('**Predicted: {}**'.format(y_pred))
