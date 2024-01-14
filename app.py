import time

import streamlit as st
from streamlit_extras.switch_page_button import switch_page
from lightGE.compoment.controller import Controller
# -- Set page config
apptitle = 'nnengine-pro'

st.set_page_config(page_title=apptitle, page_icon=":eyeglasses:")

if 'controller' not in st.session_state:
    st.session_state.controller = Controller()

if 'hyper_param_dict' not in st.session_state:
    st.session_state.hyper_param_dict = dict()


if __name__ == '__main__':
    # Title the app
    st.title(':bulb: 神经网络引擎框架')
    # 功能概览
    st.header('1 功能概览:book:',divider='rainbow')
    with st.container(border=True):
        st.markdown('这个页面详细介绍了我们框架所预先提供的模型功能，涵盖了图像分类、手写数字识别和机器翻译等多个任务。')
        flag1 = st.button(':point_up_2: 开始探索', key=1)
        if flag1:
            switch_page("Index")
    st.header('2 基本使用	:toolbox:',divider='rainbow')
    st.write('关于项目的整体流程图如下:')
    st.image('./pages/images/flow_chart.png',caption='Fig 项目流程图')
    # 模型训练
    st.subheader('2.1 模型训练:mechanical_arm:', divider='violet')
    with st.container(border=True):
        st.markdown('模型训练页面为您提供了一个全面的训练环境，您可以选择并配置适合您需求的深度学习模型'
                    '并利用丰富的数据集进行训练。通过简洁直观的界面和灵活的参数设置，您能够轻松定义模型架构、'
                    '调整超参数，并监控训练过程中的指标变化。')
        flag2 = st.button(':point_up_2: 开始训练', key=2)
        if flag2:
            switch_page("Hyper Config")
    # 模型预测
    st.subheader('2.2 模型预测:magic_wand:', divider='violet')
    with st.container(border=True):
        st.markdown('模型预测页面为您提供了一个简单易用的界面，让您能够利用训练好的深度学习模型进行实时预测和推理。'
                    '无论是图像分类、手写数字识别还是机器翻译等任务，您只需上传相应的数据，'
                    '并在页面上方选择所需的模型，即可获得快速准确的预测结果。')
        flag3 = st.button(':point_up_2: 开始预测', key=3)
        if flag3:
            switch_page("Hyper Config")


    # 帮助文档
    st.header('3 视频演示:cinema:', divider='rainbow')

    with st.container(border=True):
        st.markdown('**下面是本框架的基础训练演示**')
        train_demo = open('pages/videos/train_demo.mp4','rb')
        video_bytes = train_demo.read()
        st.video(video_bytes)

    # 帮助文档
    st.header('4 帮助文档:pencil:', divider='rainbow')
    with st.container(border=True):
        st.write('帮助文档旨在帮助用户快速上手并充分利用我们框架的功能和特性。无论是针对初学者还是有经验的用户'
                 '，帮助文档都包含了详细的步骤说明、示例代码以及常见问题解答。')
        flag4 = st.button(':point_up_2: 查看详情', key=4)
        if flag4:
            switch_page("Help")