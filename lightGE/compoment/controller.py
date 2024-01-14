from lightGE.utils.optimizer import Optimizer, SGD, OptimizerFactory
from lightGE.utils.scheduler import Scheduler, SchedulerFactory
from lightGE.utils.loss import LossFuncFactory, mseLoss
from lightGE.core.nn import Model, Linear
from lightGE.core.tensor import Tensor
from lightGE.core.transformer import Transformer
import streamlit as st
from lightGE.compoment.trainer import Trainer
from lightGE.compoment.dataloader import Dataset, ImageLoader, MnistDataset, SentenceLoader
import copy


class Controller:
    def __init__(self):
        self.hyper_parameter_config: HyperParameterConfig = HyperParameterConfig()
        self.data_config: DataConfig = DataConfig()
        self.model_config: ModelConfig = ModelConfig()
        self.task_config: TaskConfig = TaskConfig()
        self.train_dataset = Dataset()
        self.evaluate_dataset = Dataset()

        self.model = Model()
        self.sentence_loader: SentenceLoader = None

        self.trainer = None
        self.is_transformer = False

    def config_hyper_parameters(self,
                                lr: float = 0.1,
                                optimizer=None,
                                scheduler=None,
                                loss_func=None,
                                regularization_level: float = 0.1,
                                epochs: int = 100,
                                batch_size: int = 10,
                                optimizer_parameter_dict: dict = None,
                                scheduler_parameter_dict: dict = None,
                                word_vector_size: int = 512):
        self.hyper_parameter_config.config(lr=lr, optimizer=optimizer, scheduler=scheduler, loss_func=loss_func,
                                           regularization_level=regularization_level, epochs=epochs,
                                           batch_size=batch_size, optimizer_parameter_dict=optimizer_parameter_dict,
                                           scheduler_parameter_dict=scheduler_parameter_dict,
                                           word_vector_size=word_vector_size)

    def config_data_path(self, task: str, data_paths: list[str] = None):
        self.task_config.config(task)
        self.data_config.config(data_paths=data_paths)
        if self.task_config.get_task_type() == 'Image Classification':
            self.train_dataset.load_image_dir(self.data_config.get_data_path()[0],
                                              shuffle=self.data_config.get_shuffle())
            self.evaluate_dataset.load_image_dir(self.data_config.get_data_path()[1])
        elif self.task_config.get_task_type() == 'MNIST':
            mnist_dataset = MnistDataset()
            mnist_dataset.load_data(self.data_config.get_data_path()[0])
            self.train_dataset, self.evaluate_dataset = mnist_dataset.split(0.7)
        elif self.task_config.get_task_type() == 'Transformer':
            if not self.hyper_parameter_config.config_done:
                st.error('配置Transformer数据集之前请先完成超参数配置！', icon="🚨")
            else:
                self.sentence_loader = SentenceLoader(word_vector_size=self.hyper_parameter_config.get_word_vector_size())
                self.sentence_loader.load_sentences(self.data_config.get_data_path()[:2],
                                               self.data_config.get_data_path()[2:])
            self.is_transformer = True

    def config_model(self, model: Model):
        self.model_config.config(model=model)

    def prepare_trainer(self):
        if self.is_transformer:
            model = self.model_config.get_model()
            if len(model.sub_models) == 1 and isinstance(model.sub_models['0'], Transformer):
                self.model = model.sub_models['0']
            else:
                st.error('模型类型不是 Transformer，请重新配置模型', icon="🚨")
                return

        if self.__all_config_done():
            self.hyper_parameter_config.get_optimizer().set_parameters(self.model_config.get_model().parameters())

            if self.is_transformer:
                train_dataloader = (copy.deepcopy(self.sentence_loader).
                                    config(True, batch_size=self.hyper_parameter_config.get_batch_size()))
                evaluate_dataloader = (copy.deepcopy(self.sentence_loader).
                                       config(False, batch_size=self.hyper_parameter_config.get_batch_size()))
            else:
                train_dataloader = ImageLoader(batch_size=self.hyper_parameter_config.get_batch_size(),
                                               dataset=self.train_dataset)
                evaluate_dataloader = ImageLoader(batch_size=self.hyper_parameter_config.get_batch_size(),
                                                  dataset=self.evaluate_dataset)
                self.model = self.model_config.get_model()

            self.trainer = Trainer(model=self.model,
                                   optimizer=self.hyper_parameter_config.get_optimizer(),
                                   scheduler=self.hyper_parameter_config.get_scheduler(),
                                   loss_fun=self.hyper_parameter_config.get_loss_func(),
                                   epochs=self.hyper_parameter_config.get_epochs(),
                                   batch_size=self.hyper_parameter_config.get_batch_size(),
                                   train_dataloader=train_dataloader,
                                   evaluate_dataloader=evaluate_dataloader,
                                   is_transformer=self.is_transformer)

    def train(self):
        if self.trainer is None:
            st.write('trainer is not prepared!')
            return False
        self.trainer.train()
        return True

    def predict(self, x: Tensor):
        return self.model_config.get_model()(x)

    def get_label_by_index(self, index: int):
        if self.train_dataset.get_label(index) == '':
            st.error('train dataset and test dataset not configured!', icon="🚨")
        else:
            return self.train_dataset.get_label(index)

    def get_index_to_label(self):
        return self.train_dataset.index_to_label

    def __all_config_done(self):
        if not self.hyper_parameter_config.config_done:
            st.error('hyper parameters not configured completely!', icon="🚨")
            return False
        if not self.data_config.config_done:
            st.error('data paths not configured completely!', icon="🚨")
            return False
        if not self.model_config.config_done:
            st.error('model not configured completely!', icon="🚨")
            return False
        return True


class Config:
    def __init__(self):
        self.config_done = False

    def config(self):
        pass


class HyperParameterConfig(Config):
    def __init__(self):
        super(HyperParameterConfig, self).__init__()
        self.lr: float = 0.1
        self.optimizer: Optimizer = SGD([], self.lr)
        self.scheduler: Scheduler = Scheduler(self.optimizer)
        self.loss_function = mseLoss
        self.regularization_level: float = 0.1
        self.epochs: int = 100
        self.batch_size: int = 10
        self.word_vector_size: int = 512
        # self.sentence_len =

    def config(self, lr=0.1, optimizer=None, scheduler=None, loss_func=None, regularization_level=0.1, epochs=100,
               batch_size=10, optimizer_parameter_dict=None, scheduler_parameter_dict=None, word_vector_size=512):
        self.lr = lr
        if optimizer is not None:
            if isinstance(optimizer, Optimizer):
                self.optimizer = optimizer
            else:
                self.optimizer = OptimizerFactory.generate(optimizer, [], lr, optimizer_parameter_dict)
        if scheduler is not None:
            if isinstance(scheduler, Scheduler):
                self.scheduler = scheduler
            else:
                self.scheduler = SchedulerFactory.generate(scheduler, self.optimizer, scheduler_parameter_dict)
        if loss_func is not None:
            if isinstance(loss_func, str):
                self.loss_function = LossFuncFactory.generate(loss_func)
            else:
                self.loss_function = loss_func
        self.regularization_level = regularization_level
        self.epochs = epochs
        self.batch_size = batch_size

        self.scheduler.set_optimizer(self.optimizer)

        if self.optimizer.parameters is not None:
            self.config_done = True

        self.word_vector_size = word_vector_size

    def get_lr(self):
        return self.lr

    def get_optimizer(self):
        return self.optimizer

    def get_scheduler(self):
        return self.scheduler

    def get_loss_func(self):
        return self.loss_function

    def get_regularization_level(self):
        return self.regularization_level

    def get_epochs(self):
        return self.epochs

    def get_batch_size(self):
        return self.batch_size

    def get_word_vector_size(self):
        return self.word_vector_size


class DataConfig(Config):
    def __init__(self):
        super(DataConfig, self).__init__()
        self.data_paths: list[str] = []
        self.shuffle = False

    def config(self, data_paths=None, shuffle=False):
        if data_paths is None:
            data_paths = []
        self.data_paths = data_paths
        self.shuffle = shuffle

        if len(self.data_paths) != 0:
            self.config_done = True

    def get_data_path(self):
        return self.data_paths

    def get_shuffle(self):
        return self.shuffle


class ModelConfig(Config):
    def __init__(self):
        super(ModelConfig, self).__init__()
        self.model: Model = Linear(5, 5)
        self.model_save_path: str = ''

    def config(self, model=None):
        self.model = model

        if self.model is not None:
            self.config_done = True

    def get_model(self):
        return self.model


class TaskConfig(Config):  # 任务类型，一般分类任务为Classification，MNIST数据集分类任务为MNIST
    def __init__(self):
        super(TaskConfig, self).__init__()
        self.task_type = ''

    def config(self, task_type=None):
        if task_type is None:
            task_type = ''
        self.task_type = task_type

        if self.task_type != '':
            self.config_done = True

    def get_task_type(self):
        return self.task_type
