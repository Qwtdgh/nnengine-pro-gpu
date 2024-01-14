import streamlit as st
from streamlit_extras.switch_page_button import switch_page

from lightGE.compoment.controller import Controller

title = 'Parameter Config'
st.set_page_config(page_title=title, page_icon=":eyeglasses:")

if 'optimizer_parameter_dict' not in st.session_state:
    st.session_state.optimizer_parameter_dict = dict()
    st.session_state.optimizer_parameter_dict['__configured'] = ''

if 'scheduler_parameter_dict' not in st.session_state:
    st.session_state.scheduler_parameter_dict = dict()
    st.session_state.scheduler_parameter_dict['__configured'] = ''

if 'tmp_milestones' not in st.session_state:
    st.session_state.tmp_milestones = set()

if 'hyper_param_dict' not in st.session_state:
    st.session_state.hyper_param_dict = dict()

if 'controller' not in st.session_state:
    st.session_state.controller = Controller()

with st.sidebar:
    st.header('超参数设置', divider='rainbow')

    lr = st.number_input("**学习率**", format='%.6f', min_value=0.000001, value=0.000001, step=0.000001, placeholder='[请输入参数]')

    optimizer = st.selectbox("**优化器**", ('SGD', 'Adam', 'AdaGrad', 'RMSprop', 'SGDMomentum'))

    if optimizer == 'SGD':
        st.session_state.optimizer_parameter_dict['__configured'] = 'SGD'
    elif optimizer == 'Adam':
        with st.form('Adam_parameters'):
            st.write("**请选择Adam优化器的参数**")
            beta1 = st.number_input("**beta1**", format='%.6f', min_value=1e-10, max_value=1 - 1e-10, value=0.9,
                                    placeholder='[请输入参数]')
            beta2 = st.number_input("**beta2**", format='%.6f', min_value=1e-10, max_value=1 - 1e-10, value=0.98,
                                    placeholder='[请输入参数]')
            eps = st.number_input("**eps**", min_value=1e-20, value=1e-8, format='%e', placeholder='[请输入参数]')
            submitted = st.form_submit_button('确认')
            if submitted:
                st.session_state.optimizer_parameter_dict['beta'] = (beta1, beta2)
                st.session_state.optimizer_parameter_dict['eps'] = eps
                st.session_state.optimizer_parameter_dict['__configured'] = 'Adam'
            if st.session_state.optimizer_parameter_dict['__configured'] == 'Adam':
                st.success("Adam优化器的参数配置完成", icon="✅")
                st.write(f"参数如下：\nbeta={st.session_state.optimizer_parameter_dict['beta']}\n"
                         f"eps={st.session_state.optimizer_parameter_dict['eps']}")
    elif optimizer == 'AdaGrad':
        with st.form('AdaGrad_parameters'):
            st.write("**请选择AdaGrad优化器的参数**")
            eps = st.number_input("**eps**", min_value=1e-20, value=1e-8, format='%e', placeholder='[请输入参数]')
            submitted = st.form_submit_button('确认')
            if submitted:
                st.session_state.optimizer_parameter_dict['eps'] = eps
                st.session_state.optimizer_parameter_dict['__configured'] = 'AdaGrad'
            if st.session_state.optimizer_parameter_dict['__configured'] == 'AdaGrad':
                st.success("AdaGrad优化器的参数配置完成", icon="✅")
                st.write(f"参数如下：\n"
                         f"eps={st.session_state.optimizer_parameter_dict['eps']}")
    elif optimizer == 'RMSprop':
        with st.form('RMSprop_parameters'):
            st.write("**请选择RMSprop优化器的参数**")
            beta = st.number_input("**beta**", format='%.6f', min_value=1e-10, max_value=1 - 1e-10, value=0.9,
                                   placeholder='[请输入参数]')
            eps = st.number_input("**eps**", min_value=1e-20, value=1e-8, format='%e', placeholder='[请输入参数]')
            submitted = st.form_submit_button('确认')
            if submitted:
                st.session_state.optimizer_parameter_dict['beta'] = beta
                st.session_state.optimizer_parameter_dict['eps'] = eps
                st.session_state.optimizer_parameter_dict['__configured'] = 'RMSprop'
            if st.session_state.optimizer_parameter_dict['__configured'] == 'RMSprop':
                st.success("RMSprop优化器的参数配置完成", icon="✅")
                st.write(f"参数如下：\n"
                         f"beta={st.session_state.optimizer_parameter_dict['beta']}\n"
                         f"eps={st.session_state.optimizer_parameter_dict['eps']}")
    elif optimizer == 'SGDMomentum':
        with st.form('SGDMomentum_parameters'):
            st.write("**请选择SGDMomentum优化器的参数**")
            momentum = st.number_input("**momentum**", format='%.6f', min_value=1e-10, max_value=1 - 1e-10, value=0.9,
                                       placeholder='[请输入参数]')
            submitted = st.form_submit_button('确认')
            if submitted:
                st.session_state.optimizer_parameter_dict['momentum'] = momentum
                st.session_state.optimizer_parameter_dict['__configured'] = 'SGDMomentum'
            if st.session_state.optimizer_parameter_dict['__configured'] == 'SGDMomentum':
                st.success("SGDMomentum优化器的参数配置完成", icon="✅")
                st.write(f"参数如下：\n"
                         f"momentum={st.session_state.optimizer_parameter_dict['momentum']}")

    scheduler = st.selectbox("**调度器**", ('None', 'MultiStepLR', 'StepLR', 'Exponential', 'Cosine', 'ReduceLROnPlateau'))
    if scheduler == 'MultiStepLR':
        with st.form('MultiStepLR_parameters'):
            st.write("**请选择MultiStepLR调度器的参数**")
            lr_decay = st.number_input("**lr_decay**", format='%.6f', min_value=1e-10, max_value=1 - 1e-10, value=0.1,
                                       placeholder='[请输入参数]')
            milestone = st.number_input("**选择要添加的milestone**", min_value=1, value=1,
                                        placeholder='[请输入参数]')
            added = st.form_submit_button("**添加**")
            if added:
                st.session_state.tmp_milestones.add(milestone)
            st.write(f'当前选择的milestones为\n{sorted(st.session_state.tmp_milestones)}')
            submitted = st.form_submit_button('**确认**')
            if submitted:
                st.session_state.scheduler_parameter_dict['lr_decay'] = lr_decay
                st.session_state.scheduler_parameter_dict['milestones'] = sorted(st.session_state.tmp_milestones)
                st.session_state.tmp_milestones.clear()
                st.session_state.scheduler_parameter_dict['__configured'] = "MultiStepLR"
            if st.session_state.scheduler_parameter_dict['__configured'] == "MultiStepLR":
                st.success("MultiStepLR调度器的参数配置完成", icon="✅")
                st.write(f"参数如下：\n"
                         f"lr_decay={st.session_state.scheduler_parameter_dict['lr_decay']}\n"
                         f"milestones={st.session_state.scheduler_parameter_dict['milestones']}")
    elif scheduler == 'StepLR':
        with st.form('StepLR_parameters'):
            st.write("**请选择StepLR调度器的参数**")
            lr_decay = st.number_input("**lr_decay**", format='%.6f', min_value=1e-10, max_value=1 - 1e-10, value=0.1,
                                       placeholder='[请输入参数]')
            step_size = st.number_input('**step_size**', min_value=1, value=1, placeholder='[请输入参数]')
            submitted = st.form_submit_button('确认')
            if submitted:
                st.session_state.scheduler_parameter_dict['lr_decay'] = lr_decay
                st.session_state.scheduler_parameter_dict['step_size'] = step_size
                st.session_state.scheduler_parameter_dict['__configured'] = 'StepLR'
            if st.session_state.scheduler_parameter_dict['__configured'] == 'StepLR':
                st.success("StepLR调度器的参数配置完成", icon="✅")
                st.write(f"参数如下：\n"
                         f"lr_decay={st.session_state.scheduler_parameter_dict['lr_decay']}\n"
                         f"step_size={st.session_state.scheduler_parameter_dict['step_size']}")
    elif scheduler == 'Exponential':
        with st.form('Exponential_parameters'):
            st.write("**请选择Exponential调度器的参数**")
            lr_decay = st.number_input("**lr_decay**", format='%.6f', min_value=1e-10, max_value=1 - 1e-10, value=0.1,
                                       placeholder='[请输入参数]')
            submitted = st.form_submit_button('确认')
            if submitted:
                st.session_state.scheduler_parameter_dict['lr_decay'] = lr_decay
                st.session_state.scheduler_parameter_dict['__configured'] = 'Exponential'
            if st.session_state.scheduler_parameter_dict['__configured'] == 'Exponential':
                st.success("Exponential调度器的参数配置完成", icon="✅")
                st.write(f"参数如下：\n"
                         f"lr_decay={st.session_state.scheduler_parameter_dict['lr_decay']}")
    elif scheduler == 'Cosine':
        with st.form('Cosine_parameters'):
            st.write("**请选择Cosine调度器的参数**")
            T_max = st.number_input("**T_max**", min_value=1, value=1, placeholder='[请输入参数]')
            eta_min = st.number_input('**eta_min**', format='%.6f', min_value=0.0, value=0.0, placeholder='[请输入参数]')
            submitted = st.form_submit_button('确认')
            if submitted:
                st.session_state.scheduler_parameter_dict['T_max'] = T_max
                st.session_state.scheduler_parameter_dict['eta_min'] = eta_min
                st.session_state.scheduler_parameter_dict['__configured'] = 'Cosine'
            if st.session_state.scheduler_parameter_dict['__configured'] == 'Cosine':
                st.success("Cosine调度器的参数配置完成", icon="✅")
                st.write(f"参数如下：\n"
                         f"T_max={st.session_state.scheduler_parameter_dict['T_max']}\n"
                         f"eta_min={st.session_state.scheduler_parameter_dict['eta_min']}")
    elif scheduler == 'ReduceLROnPlateau':
        with st.form('ReduceLROnPlateau_parameters'):
            st.write("**请选择ReduceLROnPlateau调度器的参数**")
            lr_decay = st.number_input("**lr_decay**", format='%.6f', min_value=1e-10, max_value=1 - 1e-10, value=0.1,
                                       placeholder='[请输入参数]')
            patience = st.number_input('**patience**', min_value=1, value=1, placeholder='[请输入参数]')
            submitted = st.form_submit_button('确认')
            if submitted:
                st.session_state.scheduler_parameter_dict['lr_decay'] = lr_decay
                st.session_state.scheduler_parameter_dict['patience'] = patience
                st.session_state.scheduler_parameter_dict['__configured'] = 'ReduceLROnPlateau'
            if st.session_state.scheduler_parameter_dict['__configured'] == 'ReduceLROnPlateau':
                st.success("ReduceLROnPlateau调度器的参数配置完成", icon="✅")
                st.write(f"参数如下：\n"
                         f"lr_decay={st.session_state.scheduler_parameter_dict['lr_decay']}\n"
                         f"patience={st.session_state.scheduler_parameter_dict['patience']}")
    elif scheduler == 'None':
        st.session_state.scheduler_parameter_dict['__configured'] = 'None'

    loss_func = st.selectbox("**损失函数**", ('crossEntropyLoss', 'nll_loss', 'mseLoss', 'maeLoss', 'huberLoss',
                                              'multi_classification_cross_entropy_loss'))
    epochs = st.number_input("**训练轮次**", min_value=1, value=5, placeholder="[请输入参数]")
    batch_size = st.number_input("**批处理大小**", min_value=1, value=10, placeholder='[请输入参数]')
    word_vector_size = st.number_input("**词嵌入向量维度（针对机器翻译任务）**", min_value=1, value=512, placeholder='[请输入参数]')

    finished_configuration = st.button("**完成配置**", type='primary')
    if finished_configuration:
        if not lr:
            st.error("请配置学习率！", icon="🚨")
        elif not optimizer or optimizer != st.session_state.optimizer_parameter_dict['__configured']:
            st.error("请配置优化器！", icon="🚨")
        elif not scheduler or scheduler != st.session_state.scheduler_parameter_dict['__configured']:
            st.error("请配置调度器！", icon="🚨")
        elif not loss_func:
            st.error("请配置损失函数！", icon="🚨")
        elif not epochs:
            st.error("请配置迭代伦茨！", icon="🚨")
        elif not batch_size:
            st.error("请配置批处理大小！", icon="🚨")
        elif not word_vector_size:
            st.error("请配置词嵌入向量维度！", icon="🚨")
        else:
            st.session_state.controller.config_hyper_parameters(
                lr=lr,
                optimizer=optimizer,
                scheduler=scheduler,
                loss_func=loss_func,
                epochs=epochs,
                batch_size=batch_size,
                optimizer_parameter_dict=st.session_state.optimizer_parameter_dict,
                scheduler_parameter_dict=st.session_state.scheduler_parameter_dict,
                word_vector_size=word_vector_size)

            st.session_state.hyper_param_dict.clear()
            st.session_state.hyper_param_dict['学习率'] = lr
            st.session_state.hyper_param_dict['优化器'] = optimizer
            st.session_state.hyper_param_dict['优化器参数'] = dict()
            for key, value in st.session_state.optimizer_parameter_dict.items():
                if key != '__configured':
                    st.session_state.hyper_param_dict['优化器参数'][key] = value
            st.session_state.hyper_param_dict['调度器'] = scheduler
            st.session_state.hyper_param_dict['调度器参数'] = dict()
            for key, value in st.session_state.scheduler_parameter_dict.items():
                if key != '__configured':
                    st.session_state.hyper_param_dict['调度器参数'][key] = value
            st.session_state.hyper_param_dict['损失函数'] = loss_func
            st.session_state.hyper_param_dict['训练轮次'] = epochs
            st.session_state.hyper_param_dict['批处理大小'] = batch_size
            st.session_state.hyper_param_dict['词嵌入向量维度'] = word_vector_size

            st.session_state.optimizer_parameter_dict.clear()
            st.session_state.scheduler_parameter_dict.clear()
            st.session_state.optimizer_parameter_dict['__configured'] = ''
            st.session_state.scheduler_parameter_dict['__configured'] = ''

            st.success("超参数配置完成！", icon="✅")

st.subheader('超参数配置结果:frame_with_picture:', divider='rainbow')
with st.container(border=True):
    st.json(st.session_state.hyper_param_dict)


st.subheader('开始探索不同的机器学习任务', divider='violet')
with st.container(border=True):
    st.subheader('图像分类:frame_with_picture:')
    st.markdown(
        '图像分类是利用深度学习技术对图像进行自动分类的过程。它帮助计算机系统识别图像中的对象或场景，常见于智能相册、医学影像分析等领域。')
    flag1 = st.button(':bulb: 开始探索',key=1)
    if flag1:
        switch_page("Image Classification")

with st.container(border=True):
    st.subheader('手写数字识别:pencil2:')
    st.markdown(
        '手写数字识别利用深度学习模型将手写数字转化为可识别的数字形式。这项技术被广泛应用在验证码识别、手写笔记转文字等场景。')
    flag2 = st.button(':bulb: 开始探索',key=2)
    if flag2:
        switch_page("Mnist")

with st.container(border=True):
    st.subheader('机器翻译:slot_machine:')
    st.markdown(
        '机器翻译利用深度学习技术将一种语言的文本翻译成另一种语言。它为全球化交流、在线内容翻译等提供了便利，比如在线翻译服务和多语言内容生成。')
    flag3 = st.button(':bulb: 开始探索',key=3)
    if flag3:
        switch_page("Transformer")



