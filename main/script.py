from tqdm import tqdm
import pandas as pd
import os
from functools import partial
import numpy as np
import time
import paddle.fluid as fluid
from paddlenlp.metrics import ChunkEvaluator
import math
from collections import defaultdict
# 导入paddle库
import paddle
import paddle.nn.functional as F
import paddle.nn as nn
from paddle.io import DataLoader
from paddle.dataset.common import md5file
# 导入paddlenlp的库
import paddlenlp as ppnlp
from paddlenlp.transformers import LinearDecayWithWarmup
from paddlenlp.metrics import ChunkEvaluator
from paddlenlp.transformers import ErnieModel, ErnieTokenizer
from paddlenlp.data import Stack, Tuple, Pad, Dict
from paddlenlp.transformers import BertModel, BertTokenizer
from paddlenlp.datasets import DatasetBuilder, get_path_from_url
from paddlenlp.transformers.xlnet.modeling import XLNetModel
from paddlenlp.transformers.xlnet.tokenizer import XLNetTokenizer
from collections import Counter
# 导入所需要的py包
from paddle.io import Dataset


# 将测试集和训练集导入
with open('data/train_dataset_v2.tsv', 'r', encoding='utf-8') as handler:
    lines = handler.read().split('\n')[1:-1]
    data = list()
    for line in tqdm(lines):
        sp = line.split('\t')
        if len(sp) != 4:
            print("ERROR:", sp)
            continue
        data.append(sp)

train = pd.DataFrame(data)
train.columns = ['id', 'content', 'character', 'emotions']

test = pd.read_csv('data/test_dataset.tsv', sep='\t')
submit = pd.read_csv('data/submit_example.tsv', sep='\t')
train = train[train['emotions'] != '']
train = train.reset_index(drop=True)
len(train)

# 数据预处理
train['text'] = train['content'].astype(str) + '角色: ' + train['character'].astype(str)
test['text'] = test['content'].astype(str) + ' 角色: ' + test['character'].astype(str)

train['emotions'] = train['emotions'].apply(lambda x: [int(_i) for _i in x.split(',')])

train[['love', 'joy', 'fright', 'anger', 'fear', 'sorrow']] = train['emotions'].values.tolist()

test[['love', 'joy', 'fright', 'anger', 'fear', 'sorrow']] = [0, 0, 0, 0, 0, 0]
train.to_csv('data/train.tsv',
             columns=['id', 'content', 'character', 'text', 'love', 'joy', 'fright', 'anger', 'fear', 'sorrow'],
             sep='\t',
             index=False)

test.to_csv('data/test.tsv',
            columns=['id', 'content', 'character', 'text', 'love', 'joy', 'fright', 'anger', 'fear', 'sorrow'],
            sep='\t',
            index=False)

# 组装batch
target_cols = ['love', 'joy', 'fright', 'anger', 'fear', 'sorrow']

# 切换语言模型
PRE_TRAINED_MODEL_NAME = 'ernie-1.0'
tokenizer = ErnieTokenizer.from_pretrained('ernie-1.0')
base_model = ErnieModel.from_pretrained('ernie-1.0')


# 划分训练和测试数据
class RoleDataset(Dataset):
    def __init__(self, mode='train', trans_func=None):
        super(RoleDataset, self).__init__()
        if mode == 'train':
            self.data = pd.read_csv('data/train.tsv', sep='\t')
        else:
            self.data = pd.read_csv('data/test.tsv', sep='\t')
        self.texts = self.data['text'].tolist()
        self.labels = self.data[target_cols].to_dict('records')
        self.trans_func = trans_func

    def __getitem__(self, index):
        text = str(self.texts[index])
        label = self.labels[index]
        sample = {
            'text': text
        }
        for label_col in target_cols:
            sample[label_col] = label[label_col]
        sample = self.trans_func(sample)
        return sample

    def __len__(self):
        return len(self.texts)


# 转换成id的函数
# 调用tokenizer的数据处理方法把文本转为id,得到编码结果作为输入
def convert_example(example, tokenizer, max_len=512, is_test=False):
    # print(example)
    sample = {}
    encoded_inputs = tokenizer(text=example["text"], max_len=max_len)
    sample['input_ids'] = encoded_inputs["input_ids"]
    sample['token_type_ids'] = encoded_inputs["token_type_ids"]

    sample['love'] = np.array(example["love"], dtype="float32")
    sample['joy'] = np.array(example["joy"], dtype="float32")
    sample['anger'] = np.array(example["anger"], dtype="float32")

    sample['fright'] = np.array(example["fright"], dtype="float32")
    sample['fear'] = np.array(example["fear"], dtype="float32")
    sample['sorrow'] = np.array(example["sorrow"], dtype="float32")

    return sample


max_len = 128
trans_func = partial(
    convert_example,
    tokenizer=tokenizer,
    max_len=max_len)
trans_func
train_ds = RoleDataset('train', trans_func)
test_ds = RoleDataset('test', trans_func)


# 创建迭代器函数
#使用BatchSample对数据进行切分 DistributedBatchSampler支持多卡并行训练
def Dataloader(dataset,
                      mode='train',
                      batch_size=1,
                      batchify_fn=None):
    shuffle = True if mode == 'train' else False
    if mode == 'train':
        batch_sampler = paddle.io.DistributedBatchSampler(
            dataset, batch_size=batch_size, shuffle=shuffle)
    else:
        batch_sampler = paddle.io.BatchSampler(
            dataset, batch_size=batch_size, shuffle=shuffle)

    return paddle.io.DataLoader(
        dataset=dataset,
        batch_sampler=batch_sampler,
        collate_fn=batchify_fn,
        return_list=True)


# 对每一个batch数据进行处理
    # 获取batch数据的大小
    # 如果batch_size为0，则返回一个空字典
    # 遍历batch数据，将每一个数据，转换成tensor的形式
# #对一个batch内的数据，进行pad
#     # 函数Stack具有相同维度数据构建一个batch（数据维度必须相同）
#     # 函数Pad将不同长度的句子统一长度，统一长度大小以最大为标准。参数pad_val指定补齐时，用什么。参数axis=0按列，axis=1按列
def collate_func(batch_data):
    batch_size = len(batch_data)
    if batch_size == 0:
        return {}
    input_ids_list, attention_mask_list = [], []
    love_list, joy_list, anger_list = [], [], []
    fright_list, fear_list, sorrow_list = [], [], []
    for instance in batch_data:
        input_ids_temp = instance["input_ids"]
        attention_mask_temp = instance["token_type_ids"]
        love = instance['love']
        joy = instance['joy']
        anger = instance['anger']
        fright = instance['fright']
        fear = instance['fear']
        sorrow = instance['sorrow']
        input_ids_list.append(paddle.to_tensor(input_ids_temp, dtype="int64"))
        attention_mask_list.append(paddle.to_tensor(attention_mask_temp, dtype="int64"))
        love_list.append(love)
        joy_list.append(joy)
        anger_list.append(anger)
        fright_list.append(fright)
        fear_list.append(fear)
        sorrow_list.append(sorrow)
    return {"input_ids": Pad(pad_val=0, axis=0)(input_ids_list),
            "token_type_ids": Pad(pad_val=0, axis=0)(attention_mask_list),
            "love": Stack(dtype="int64")(love_list),
            "joy": Stack(dtype="int64")(joy_list),
            "anger": Stack(dtype="int64")(anger_list),
            "fright": Stack(dtype="int64")(fright_list),
            "fear": Stack(dtype="int64")(fear_list),
            "sorrow": Stack(dtype="int64")(sorrow_list),
            }


# 构建分类器
class EmotionClassifier(nn.Layer):
    def __init__(self, bert, n_classes):
        super(EmotionClassifier, self).__init__()
        self.bert = bert
        self.out_love = nn.Linear(self.bert.config["hidden_size"], n_classes)
        self.out_joy = nn.Linear(self.bert.config["hidden_size"], n_classes)
        self.out_fright = nn.Linear(self.bert.config["hidden_size"], n_classes)
        self.out_anger = nn.Linear(self.bert.config["hidden_size"], n_classes)
        self.out_fear = nn.Linear(self.bert.config["hidden_size"], n_classes)
        self.out_sorrow = nn.Linear(self.bert.config["hidden_size"], n_classes)

    def forward(self, input_ids, token_type_ids):
        _, pooled_output = self.bert(
            input_ids=input_ids,
            token_type_ids=token_type_ids
        )
        love = self.out_love(pooled_output)
        joy = self.out_joy(pooled_output)
        fright = self.out_fright(pooled_output)
        anger = self.out_anger(pooled_output)
        fear = self.out_fear(pooled_output)
        sorrow = self.out_sorrow(pooled_output)
        return {
            'love': love, 'joy': joy, 'fright': fright,
            'anger': anger, 'fear': fear, 'sorrow': sorrow,
        }


#class_names = [1]

# 定义超参数(使用网格搜索寻找到较好的参数)
EPOCHS = [3]
weight_decay = 1e-4
data_path = 'data'
warmup_proportion = 0.0
init_from_ckpt = None
batch_sizes = [32]

learning_rates = [3e-5]


# 模型训练
def evaluate(model, data_loader, criterion, optimizer, scheduler, metric):
    model.train()
    global_step = 0
    tic_train = time.time()
    log_steps = 100
    for epoch in range(EPOCHS):
        losses = []
        for step, sample in enumerate(data_loader):
            input_ids = sample["input_ids"]
            token_type_ids = sample["token_type_ids"]
            outputs = model(input_ids=input_ids,
                            token_type_ids=token_type_ids)
            loss_love = criterion(outputs['love'], sample['love'])
            loss_joy = criterion(outputs['joy'], sample['joy'])
            loss_fright = criterion(outputs['fright'], sample['fright'])
            loss_anger = criterion(outputs['anger'], sample['anger'])
            loss_fear = criterion(outputs['fear'], sample['fear'])
            loss_sorrow = criterion(outputs['sorrow'], sample['sorrow'])
            loss = loss_love + loss_joy + loss_fright + loss_anger + loss_fear + loss_sorrow

            for label_col in target_cols:
                correct = metric.compute(outputs[label_col], sample[label_col])
                metric.update(correct)
            acc = metric.accumulate()
            losses.append(loss.numpy())
            loss.backward()
            global_step += 1
            # 每间隔 log_steps 输出训练指标
            if global_step % log_steps == 0:
                print("global step %d, epoch: %d, batch: %d, loss: %.5f, accuracy: %.5f, speed: %.2f step/s"
                      % (global_step, epoch, step, loss, acc,log_steps / (time.time() - tic_train)))

            optimizer.step()
            scheduler.step()
            optimizer.clear_grad()

        metric.reset()
    print("loss: ", np.mean(losses))


# 模型预测
def predict(model, test_data_loader):
    val_loss = 0
    test_pred = defaultdict(list)
    model.eval()
    for step, batch in (enumerate(test_data_loader)):
        b_input_ids = batch['input_ids']
        token_type_ids = batch['token_type_ids']

        with paddle.no_grad():
            logits = model(input_ids=b_input_ids, token_type_ids=token_type_ids)
            for col in target_cols:
                out2 = paddle.argmax(logits[col], axis=1)
                test_pred[col].extend(out2.numpy().tolist())
    return test_pred


# 训练以及测试
for learning_rate in learning_rates:
    for epochs in EPOCHS:
        for batch_size in batch_sizes:
            print('lr:', learning_rate, ' epochs:', epochs, ' batch_size:', batch_size)
            model = EmotionClassifier(base_model, 4)
            train_data_loader = Dataloader(
                train_ds,
                mode='train',
                batch_size=batch_size,
                batchify_fn=collate_func)

            EPOCHS = 3
            num_training_steps = len(train_data_loader) * EPOCHS

            # 定义 learning_rate_scheduler，负责在训练过程中对 lr 进行调度
            lr_scheduler = ppnlp.transformers.CosineDecayWithWarmup(learning_rate, num_training_steps,
                                                                    warmup_proportion)

            # Generate parameter names needed to perform weight decay.
            # All bias and LayerNorm parameters are excluded.
            decay_params = [
                p.name for n, p in model.named_parameters()
                if not any(nd in n for nd in ["bias", "norm"])
            ]

            # 定义 Optimizer
            optimizer = paddle.optimizer.AdamW(
                learning_rate=lr_scheduler,
                parameters=model.parameters(),
                weight_decay=0.0,
                apply_decay_param_fun=lambda x: x in decay_params)
            # 交叉熵损失
            criterion = paddle.nn.CrossEntropyLoss()
            # 评估的时候采用准确率指标
            metric = paddle.metric.Accuracy()

            evaluate(model, train_data_loader, criterion, optimizer, lr_scheduler, metric)

            test_data_loader = Dataloader(
                test_ds,
                mode='test',
                batch_size=batch_size,
                batchify_fn=collate_func)

            test_pred = predict(model, test_data_loader)


# 预测结果输出
label_preds = []
for col in target_cols:
    preds = test_pred[col]
    label_preds.append(preds)
print(len(label_preds[0]))
sub = test['text']
sub = pd.DataFrame(sub)
sub['emotion'] = np.stack(label_preds, axis=1).tolist()
sub['emotion'] = sub['emotion'].apply(lambda x: ','.join([str(i) for i in x]))
sub.to_csv('result.tsv'.format(PRE_TRAINED_MODEL_NAME), sep='\t', index=False)
sub.head()

