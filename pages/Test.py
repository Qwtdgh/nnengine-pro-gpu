
import streamlit as st
from lightGE.data import Dataset, DataLoader
from lightGE.utils import OptimizerFactory, LossFuncFactory
from lightGE.utils.scheduler import MultiStepLR
from streamlit_echarts import st_echarts, JsCode
import numpy as np

from lightGE.utils import Trainer

import logging

# -- Set page config
pagetitle = 'nnengine-Test'

st.set_page_config(page_title=pagetitle, page_icon=":eyeglasses:")

if 'model_repr' not in st.session_state:
    st.session_state.model_repr = ''

if 'model' not in st.session_state:
    st.session_state.model = None

# 创建一个空列表来存储日志消息
log_messages = []


# 创建一个StreamHandler来捕获日志消息
class StreamlitHandler(logging.Handler):
    def emit(self, record):
        msg = self.format(record)
        log_messages.append(msg)  # 将日志消息添加到列表中


# 创建日志记录器
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# 将StreamlitHandler添加到日志记录器
logger.addHandler(StreamlitHandler())

loss = ''
start = False
save_model = False
end = False
if 'download' not in st.session_state:
    st.session_state.download = False


def config(m, epoch: int, batch: int, optimizer, loss_func):

    # 随机100*2大小的数据服从标准正态分布
    data = np.random.randn(100, 2)

    # 生成数据对应的标签
    labels = data[:, 0:1] + 10 * data[:, 1:2]

    # 生成数据集
    dataset = Dataset(data, labels)

    # 分割训练集 和 测试集
    train_dataset, test_dataset = dataset.split(0.8)

    # 优化器
    opt = OptimizerFactory().generate(name=optimizer, parameters=m.parameters(), lr=0.001)
    print(opt.lr)
    loss_f = LossFuncFactory().generate(loss_func)
    sch = MultiStepLR(opt, [10, 20, 30, 40, 50, 60, 70, 80, 90])

    trainer = Trainer(m, opt, loss_f, {
        "epochs": epoch,
        "batch_size": batch,
        "shuffle": True,
        "save_path": "./tmp/model.pkl"
    }, sch)
    # 计算损失
    min_loss = trainer.train(DataLoader(train_dataset, batch), DataLoader(test_dataset, batch))
    return trainer, min_loss


def render_basic_line_chart(x_arr, arr_train_loss, arr_eval_loss):
    option = {
        "legend": {
            "right": '10%'
        },
        "xAxis": {
            "type": 'category',
            "data": x_arr
        },
        "yAxis": {"type": "value"},
        "tooltip": {
            "formatter": JsCode("function (params) \
                { return `${params.seriesName}<br />epoch ${params.name}：${params.value.toFixed(5)}` ;}").js_code
        },
        "series": [{
            "data": arr_train_loss,
            "type": "line",
            "name": "train loss",
            # "label": {
            #     "show": True,
            #     "position": 'top',
            #     "formatter": JsCode("function (params){ return params.data.toFixed(3);}").js_code
            # },
        }, {
            "data": arr_eval_loss,
            "type": "line",
            "name": "eval loss",
        }
        ]}
    st_echarts(
        options=option, height="400px",
    )


def render_basic_area_chart(x_arr, arr_lr):
    options = {
        "xAxis": {
            "type": "category",
            "boundaryGap": False,
            "data": x_arr
        },
        "yAxis": {"type": "value"},
        "series": [
            {
                "data": arr_lr,
                "type": "line",
                "areaStyle": {},
            }
        ],
    }
    st_echarts(options=options)


def click_download():
    st.session_state.download = True


with st.sidebar:
    epoch = st.number_input("**选择epochs大小:**", value=10, placeholder="Type a number...")
    batch = st.number_input("**选择batch大小:**", value=10, placeholder="Type a number...")
    optimizer = st.selectbox('**选择优化器:**', ('SGD', 'Adam', 'AdaGrad', 'RMSprop', 'SGDMomentum'),
                             help='这里是一些建议')
    loss_func = st.selectbox('**选择损失函数:**', ('mseLoss', 'maeLoss', 'crossEntropyLoss', 'huberLoss', 'nll_loss'),
                             help='这里是一些建议')

    save_model = st.checkbox('*Save the model*')
    checked = st.button(':sparkles: :rainbow[Start Train]')

st.title("模型训练")
st.subheader('当前模型', divider='blue')
st.code(st.session_state.model_repr)
m = st.session_state.model

if checked:
    start = True
    trainer, loss = config(m, epoch, batch, optimizer, loss_func)
    end = True

if start:
    st.divider()
    st.subheader("loss曲线")
    x_arr = [i + 1 for i in range(epoch)]
    render_basic_line_chart(x_arr, trainer.arr_train_loss, trainer.arr_eval_loss)
    st.subheader("lr曲线")
    render_basic_area_chart(x_arr, trainer.arr_lr)
    end = True

if save_model and end:
    st.divider()
    st.subheader('保存模型')
    with open('./tmp/model.pkl', 'rb') as file:
        download = st.download_button(
            label="Download model as PKL",
            data=file,
            file_name='model.pkl', on_click=click_download
        )
if st.session_state.download:
    st.balloons()
    save_model = False
st.session_state.download = False
# arr_loss = np.vstack((trainer.arr_train_loss, trainer.arr_eval_loss)).transpose()
# row_array = pd.Series([1, 2, 3, 4, 5, 6])
# chart_data = pd.DataFrame(arr_loss, columns=["train_loss", "eval_loss"])
# loss_line = st.line_chart(chart_data)
