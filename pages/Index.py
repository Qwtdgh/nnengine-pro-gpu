
import streamlit as st
from streamlit_extras.switch_page_button import switch_page

title = 'Index'
st.set_page_config(page_title=title, page_icon=":eyeglasses:")

st.title("功能概览")
# 图片列表
task1_path = 'pages/images/task_classify.png'
task1_caption = 'Fig1 图像分类'

task2_path = 'pages/images/task_digit.png'
task2_caption = 'Fig2 手写数字识别'

task3_path = 'pages/images/task_translation.png'
task3_caption = 'Fig3 机器翻译'

# 介绍

st.subheader('图像分类:frame_with_picture:', divider='violet')
st.image(task1_path, task1_caption)
with st.container(border=True):
    st.markdown(
        '图像分类是利用深度学习技术对图像进行自动分类的过程。它帮助计算机系统识别图像中的对象或场景，常见于智能相册、医学影像分析等领域。')
    flag1 = st.button(':bulb: 开始探索',key=1)
    if flag1:
        # TODO predict
        switch_page("Image Classification")
st.subheader('手写数字识别:pencil2:', divider='violet')
st.image(task2_path, task2_caption)
with st.container(border=True):
    st.markdown(
        '手写数字识别利用深度学习模型将手写数字转化为可识别的数字形式。这项技术被广泛应用在验证码识别、手写笔记转文字等场景。')
    flag2 = st.button(':bulb: 开始探索',key=2)
    if flag2:
        # TODO predict
        switch_page("Mnist")
st.subheader('机器翻译:slot_machine:', divider='violet')
st.image(task3_path, task3_caption)
with st.container(border=True):
    st.markdown(
        '机器翻译利用深度学习技术将一种语言的文本翻译成另一种语言。它为全球化交流、在线内容翻译等提供了便利，比如在线翻译服务和多语言内容生成。')
    flag3 = st.button(':bulb: 开始探索',key=3)
    if flag3:
        # TODO predict
        switch_page("Transformer")