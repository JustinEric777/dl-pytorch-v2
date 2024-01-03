import sys

sys.path.append('../6. rnn')

from dataset_and_model import load_data_time_machine

# 长短期记忆网络
# 门控记忆元：
# 输入门：何时将数据读入单元
# 忘记门：重置单元的内容
# 输出门：从单元中输出条目
#


# load 数据
batch_size, num_steps = 32, 35
train_iter, vocab = load_data_time_machine(batch_size, num_steps)

if __name__ == "__main__":
    print(train_iter)
    print(vocab)
