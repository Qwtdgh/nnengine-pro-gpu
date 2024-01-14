import streamlit as st


title = 'Help'
st.set_page_config(page_title=title, page_icon=":eyeglasses:")

if __name__ == '__main__':
    st.header('帮助文档:pencil:')
    st.subheader('1 总体架构',divider='rainbow')
    st.image('https://typoraqlh.oss-cn-beijing.aliyuncs.com/qlh/typora/image-20231206092147180.png', 'Fig1 总体架构图')
    st.markdown("""nnengine框架可分为前端、后端两部分，其中

* **用户层** 指定系统面向的使用人员，分为普通用户和模型开发者。

* **表现层** 系统向用户提供的功能。用户通过在前端界面操作，即可实现配置任务、执行训练任务、展示评估结果、使用模型预测等功能。

* **交互层** 用于实现前后端交互，将前端配置的任务信息传递给后端，并将后端中任务执行的结果返回给前端展示。接口分为任务配置接口、任务执行接口、数据上传接口和模型下载接口等。由于我们采用的 Streamlit 结构的特性，前后端实际上是通过共享数据来进行交互的。在这种情况下，传统的前后端数据交换机制并不适用，因此没有设计用于交互的组件。

* **组件层** 将后端服务划分成模型控制组件、数据集加载组件、模型训练组件和模型预测组件。并由模型控制组件控制其余三个组件的交互，使系统实现高内聚低耦合。

* **模型层** 后端中基本的类对象，实现模型训练、预测等基本功能，支持Transformer模型和处理sentence类型的数据集，并提供多种损失函数和优化器等。

* **数据层** 用于保存后端中模型训练、测试的数据集，以及训练后得到的模型和评估结果等。
    """)
    st.subheader('2 帮助建议',divider='rainbow')
    st.markdown('#### 2.1 训练调优建议')
    st.markdown('''
    1. **学习率调整：** 尝试不同的学习率，例如使用学习率衰减策略或使用动态调整学习率的优化器，以便更好地收敛模型。
2. **批量大小优化：** 调整批量大小可能有助于提高训练效率和模型性能。尝试不同的批量大小，以找到最适合数据集和模型的大小。
3. **数据增强技术：** 使用数据增强技术（如旋转、翻转、缩放等）可以扩增数据集，提高模型的泛化能力和稳健性。
4. **正则化方法：** 考虑使用正则化技术（如Dropout、L1/L2正则化）来减少过拟合情况，提高模型的泛化性能。
    ''')
    st.markdown('#### 2.2 模型优化建议')
    st.markdown('''
    1. **迁移学习：** 尝试使用预训练的模型作为基础，并在其基础上微调以适应特定任务，可以有效提高模型性能。
2. **模型复杂度控制：** 在保证模型性能的前提下，控制模型的复杂度，避免过度拟合。
3. **特征工程：** 对于特定任务，设计有效的特征工程可以提高模型的表现，尤其是在数据量较少或特征较为复杂时。
4. **超参数优化：** 通过使用自动化的超参数优化工具（如网格搜索、贝叶斯优化等）寻找最佳的超参数组合，以提高模型性能。

这些训练调优和模型优化建议可以帮助您优化模型性能并提高训练效率，但具体的优化方法需要根据具体任务和数据集来调整和选择。
    ''')

    st.subheader('3 步骤说明',divider='rainbow')
    st.markdown('''
    #### 1. 图像分类

步骤：图像分类任务是将图像分配到特定类别的过程，通常包括数据准备、模型选择和训练配置等步骤。例如，数据准备可能包括数据集预处理和分割，模型选择可能根据任务需求选择合适的卷积神经网络模型，训练配置可能包括学习率、批量大小等超参数的设定。

#### 2. 手写数字识别

步骤：手写数字识别是通过深度学习模型识别手写数字的过程。数据集准备包括手写数字图像收集和标记，模型选择可能基于卷积神经网络或者其他适合分类任务的模型，训练过程包括对模型的训练、验证和调优。

#### 3. 机器翻译

步骤：机器翻译是将一种语言的文本翻译成另一种语言的任务。在这个过程中，数据集的选择和准备至关重要，同时需要选择适合翻译任务的模型，并进行相应的配置和训练过程。
    ''')
    st.subheader('示例代码')
    st.markdown('下面的代码展示了如何训练手写识别数字集Mnist。')
    st.code('''
    import streamlit as st
from lightGE.core import MNIST

title = 'Mnist Train'
st.set_page_config(page_title=title, page_icon=":eyeglasses:")

with st.sidebar:
    st.header('数据集加载', divider='rainbow')
    dataDir = st.text_input('**数据路径：**', value="./res/mnist/", placeholder="Type dataset path...")
    check_train = st.button(':sparkles: :rainbow[开始训练]')

st.title("MNIST数据集分类任务")

if check_train:
    model = MNIST()
    cache_path = 'tmp/mnist.pkl'
    controller = st.session_state.controller
    controller.config_path(data_paths=[dataDir])
    controller.config_model(model=model, save_path=cache_path)
    controller.config_task("MNIST")
    controller.train()

    ''',language='python')

    st.header('4 Q&A',divider='rainbow')
    with st.container(border=True):
        st.markdown('''
    **Q: 框架提供了哪些主要功能？**
    
    A: 我们的框架提供了一系列深度学习任务的支持，包括图像分类、手写数字识别和机器翻译等。图像分类任务涉及将图像分为不同类别，手写数字识别可识别手写数字图像，机器翻译则是将一种语言的文本翻译成另一种语言。
    
    **Q: 这个帮助文档适合哪些用户？**
    
     A: 此文档旨在服务广泛的用户群体，从初学者到有一定经验的深度学习从业者。对于新手，文档提供了基础入门指南；对于有经验者，文档则提供了更深层次的高级功能使用方法和优化建议。
    
    **Q: 训练过程中出现收敛问题怎么办？**
    
     A: 收敛问题可能由学习率过高、模型复杂度等因素导致。可以尝试降低学习率、增加数据量、使用正则化技术等方法。
    
    **Q: 如何优化模型的准确性？**
    
     A: 模型准确性的优化需要考虑数据增强、模型选择、超参数调整等多个方面。适当的数据增强和超参数调整可以提升模型的性能。
        ''')

    st.header('5 联系我们',divider='rainbow')
    st.markdown('''
    如果您在使用我们框架的过程中遇到任何问题、有建议或者需要进一步的帮助，请随时联系我们的团队。您可以通过以下方式与我们取得联系：

- **邮箱：** 
  - haohe@buaa.edu.cn
  - 2775257495@qq.com
  - 2211676811@qq.com
  - lian_hongrui@buaa.com
  - zy2306408@buaa.edu.cn
  - yux_10@qq.com

​									**——项目来自于吴际老师的《高等软件工程》**

我们的团队将竭诚为您提供支持，并欢迎您的反馈和建议，以便我们不断改进和完善我们的框架。感谢您选择使用我们的产品！
    ''')