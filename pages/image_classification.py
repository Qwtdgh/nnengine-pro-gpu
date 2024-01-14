import os
import pickle
import shutil
import numpy as np
import streamlit as st

import zipfile

from lightGE.core.tensor import Tensor
from lightGE.compoment.controller import Controller

from PIL import Image

if 'controller' not in st.session_state:
    st.session_state.controller = Controller()

if 'classification_training_set_dir' not in st.session_state:
    st.session_state.classification_training_set_dir = ''

if 'classification_test_set_dir' not in st.session_state:
    st.session_state.classification_test_set_dir = ''

if 'classification_model_path' not in st.session_state:
    st.session_state.classification_model_path = ''

if 'hyper_param_dict' not in st.session_state:
    st.session_state.hyper_param_dict = dict()

if 'model_repr' not in st.session_state:
    st.session_state.model_repr = ''

if 'classification_training_set_uploaded' not in st.session_state:
    st.session_state.classification_training_set_uploaded = False

if 'classification_test_set_uploaded' not in st.session_state:
    st.session_state.classification_test_set_uploaded = False

if 'classification_image_uploaded' not in st.session_state:
    st.session_state.classification_image_uploaded = False


def training_set_uploaded():
    st.session_state.classification_training_set_uploaded = True


def test_set_uploaded():
    st.session_state.classification_test_set_uploaded = True


def image_uploaded():
    st.session_state.classification_image_uploaded = True


with st.sidebar:
    st.subheader('当前选择模型', divider='blue')
    st.code(st.session_state.model_repr)
    st.subheader('当前超参数配置', divider='blue')
    with st.container(border=True):
        st.json(st.session_state.hyper_param_dict)

st.title("Image Classification")
train_tab, prediction_tab = st.tabs(['图像分类任务-模型训练', '图像分类任务-模型预测'])

with train_tab:
    with st.container(border=True):
        st.write('**图像数据集选择**')
        st.write('> 请保证选择的数据集为一个文件夹，该文件夹具有如下的结构')

        dir_example = dict()
        dir_example['training_set_directory'] = dict()
        dir_example['training_set_directory']['type1_directory'] = ['type1_image1.png', 'type1_image2.png',
                                                                    'type1_image3.png', '...']
        dir_example['training_set_directory']['type2_directory'] = ['type2_image1.png', 'type2_image2.png',
                                                                    'type2_image3.png', '...']
        dir_example['training_set_directory']['type3_directory'] = ['type3_image1.png', 'type3_image2.png',
                                                                    'type3_image3.png', '...']

        st.json(dir_example, expanded=False)

        uploaded_training_file = st.file_uploader('选择训练集zip文件', type='.zip', on_change=training_set_uploaded)
        if st.session_state.classification_training_set_uploaded and uploaded_training_file is not None:
            with st.status('zip文件处理中...'):
                st.write('读取zip文件中...')
                zip_file = zipfile.ZipFile(uploaded_training_file)
                st.write('读取zip文件完成✅')
                st.write('解压zip文件...')
                new_dir = 'res/zip_unpack/training_set'
                dir_name = zip_file.namelist()[0]
                if os.path.exists(new_dir):
                    shutil.rmtree(new_dir)
                os.makedirs(new_dir)
                zip_file.extractall(new_dir)
                st.session_state.classification_training_set_dir = os.path.join(new_dir, dir_name)
                if st.session_state.classification_training_set_dir != '' \
                        and st.session_state.classification_test_set_dir != '':
                    st.session_state.controller.config_data_path('Image Classification',
                                                                 [st.session_state.classification_training_set_dir,
                                                                  st.session_state.classification_test_set_dir])
                st.write('解压zip文件完成✅')
            st.session_state.classification_training_set_uploaded = False

        uploaded_test_file = st.file_uploader('选择测试集zip文件', type='.zip', on_change=test_set_uploaded)
        if st.session_state.classification_test_set_uploaded and uploaded_test_file is not None:
            with st.status('zip文件处理中...'):
                st.write('读取zip文件...')
                zip_file = zipfile.ZipFile(uploaded_test_file)
                st.write('读取zip文件完成✅')
                st.write('解压zip文件...')
                new_dir = 'res/zip_unpack/test_set'
                dir_name = zip_file.namelist()[0]
                if os.path.exists(new_dir):
                    shutil.rmtree(new_dir)
                os.makedirs(new_dir)
                zip_file.extractall(new_dir)
                st.session_state.classification_test_set_dir = os.path.join(new_dir, dir_name)
                if st.session_state.classification_training_set_dir != '' \
                        and st.session_state.classification_test_set_dir != '':
                    st.session_state.controller.config_data_path('Image Classification',
                                                                 [st.session_state.classification_training_set_dir,
                                                                  st.session_state.classification_test_set_dir])
                st.write('解压zip文件完成✅')
            st.session_state.classification_test_set_uploaded = False

    types_dict = st.session_state.controller.get_index_to_label()
    with st.container(border=True):
        st.write(f'当前数据集的类别数：{len(types_dict)}')
        st.json(types_dict)

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
    uploaded_image_file = st.file_uploader('选择待预测的图片', type=["png", "jpg", "jpeg"], on_change=image_uploaded)
    if st.session_state.classification_image_uploaded and uploaded_image_file is not None:
        with st.status('图片处理中...'):
            st.write('图片处理中...')
            image = Image.open(uploaded_image_file)
            img_array = np.array([np.array(image) / 255.0])  # 添加额外的批处理维度
            img_array = img_array.transpose((0, 3, 1, 2))  # 改变维度顺序为 B C H W
            img_tensor = Tensor(img_array)
            y_pred = st.session_state.controller.predict(img_tensor)
            predicted_type = st.session_state.controller.get_label_by_index(np.argmax(y_pred.data, axis=1)[0])
            st.write('图片处理完成✅')
        col_pic, col_res= st.columns(2)
        with col_pic:
            con1 = st.container(border=True)
            with con1:
                st.header("图片内容")
                # 显示上传的图片
                st.image(image, caption="上传的图片", use_column_width=True)
        with col_res:
            # 显示预测结果
            con2 = st.container(border=True)
            with con2:
                st.header("预测结果")
                st.markdown('**预测结果: {}**'.format(predicted_type))
            # 显示概率
                st.header('预测概率')
                pro_dict = {}
                for i in range(y_pred.shape[1]):
                    label = st.session_state.controller.get_label_by_index(i)
                    pro_dict[label] = y_pred.data[0][i]
                st.json(pro_dict)
        st.session_state.classification_image_uploaded = False
