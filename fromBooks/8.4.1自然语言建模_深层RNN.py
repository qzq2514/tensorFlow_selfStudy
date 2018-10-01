import tensorflow as tf
import numpy as np
from tensorflow.models.rnn.ptb import reader

DATA_PATH="ptbData/simple-examples/data"

HIDDEN_SIZE=200 #一个LSTM单元中每个神经网络单元的隐藏层神经元个数,也是每个时刻每个样本的状态向量的长度，
                #因为(本时刻输入+上一时刻状态)经过隐藏层后得到的就是下一时刻的状态，（一个LSTM有4个神经网络单元，详见“LSTM单元结构图.png”）
NUM_LAYERS=2   #深层RNN的层数
VOCAB_SIZE=10000   #词典规模，共有10000个单词

LEARNING_RATE=1.0
TRAIN_BATCH_SIZE=20   #测试数据batch大小，batch中每个元素是一句话，每句话由多个用ID表示的单词构成，这里表示每个batch中有20个短句/短语
TRAIN_NUM_STEP=35     #每句话的长度，即单词数(又称截断长度)

EVAL_BATCH_SIZE=1
EVAL_NUM_STEP=1

NUM_EPOCH=2
KEEP_PROB=0.5
MAX_GARD_NORM=5     #用于控制梯度膨胀系数，防止梯度消失

class PTBModel(object):
    def __init__(self,is_training,batch_size,num_steps):

        self.batch_size=batch_size
        self.num_steps=num_steps


        #每批有batch_size个样本，每个样本包括num_steps个序列单词
        #对应训练集中就是每批有TRAIN_BATCH_SIZE=20个句子，每个句子TRAIN_NUM_STEP=35个单词
        self.input_data=tf.placeholder(tf.int32,[batch_size,num_steps])
        #预测输出和样本输入的尺寸一样，因为输入句子作为输入，其对应的标签就是句中每个单词的下一个单词
        self.targets=tf.placeholder(tf.int32,[batch_size,num_steps])

        #定义一层lstm（其实就是定义一个LSTM单元）,每个LSTM中的神经网络单元有HIDDEN_SIZE个神经元
        lstm_cell=tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE)

        # print("state_size:",lstm_cell.state_size)  # state_size为隐藏层单元，也是每一时刻的状态的长度，详见"8.1循环神经网络前向传播示意图.png"

        if is_training:
            lstm_cell=tf.nn.rnn_cell.DropoutWrapper(
                lstm_cell,output_keep_prob=KEEP_PROB)
        # cell = tf.nn.rnn_cell.MultiRNNCell([get_a_cell() for _ in range(NUM_LAYERS)])  #不要用这种
        cell=tf.nn.rnn_cell.MultiRNNCell([lstm_cell]*NUM_LAYERS)  #定义多层lstm构成一个深层RNN

        self.initial_state = cell.zero_state(batch_size, tf.float32)
        # print("self.initial_state:",type(self.initial_state))
        # self.initial_state=cell.zero_state(batch_size,tf.float32)  #每个样本给定一个初始0状态
        # print("len:",len(self.initial_state))                #类型为tuple,长度为2,包含两个矩阵，每一层的RNN有一个状态
        # print(self.initial_state)                            #每个矩阵的大小shanpe是[20,200],即每层的状态为[20,200]
                                                            #20对应batch_size，200对应隐藏层的神经单元个数

        #将整个单词表内所有单词的ID转换成单词向量，词典中共有VOCAB_SIZE,每个单词变成长度为HIDDEN_SIZE的向量
        embedding=tf.get_variable("embedding",[VOCAB_SIZE,HIDDEN_SIZE])    #embedding词向量矩阵大小为(10000, 200)

        # with tf.Session() as sess:
        #     sess.run(tf.global_variables_initializer())      #根据名称"embedding"就得到了已经在RNN模型中定义好的单词矩阵
        #     print(sess.run(embedding))

        # 根据单词的向量表embedding，在表中找到self.input_data对应索引的向量
        # 将原本batch_size*num_steps个单词ID转化为单词向量，转化后输入层的维度为batch_size*num_steps*HIDDEN_SIZE
        inputs=tf.nn.embedding_lookup(embedding,self.input_data)  #shape=(20, 35, 200)
        # print(inputs)

        if is_training:
            inputs=tf.nn.dropout(inputs,KEEP_PROB) #只在训练时使用dropout

        # print(inputs[:,2,:])   #(20, 200)
        outputs=[]
        state=self.initial_state    #state存储不同样本的状态
        with tf.variable_scope("RNN"):
            for time_step in range(num_steps):     #num_steps就是每句话的单词个数
                if time_step>0:
                    tf.get_variable_scope().reuse_variables()   #允许共享当前scope下的所有变量
                #经过cell必须是某一时刻的输入，不能将所有时刻全都放进cell，所以这里经过inputs[:,time_step,:]将batch中每一时刻的输入提取出来放进cell中
                #batch大小是20个样本，每个样本中的每个时刻的输入是一个单词(这里用长度为200的embedding向量表示一个单词)
                cell_out,state=cell(inputs[:,time_step,:],state)  #从输入数据获得当前时刻输入并传入LSTM结构,也可以cell.__call__(xx, xx)
                #print(cell_out)    #shape=(20, 200)
                #print(state)      #shape=(20, 200),经过RNN或者LSTM单元后得到的下一时刻状态和当前输出的大小是一样的
                outputs.append(cell_out)     #将当前输出加入输出队列

        # print("outputs：",outputs)      #outputs是一个长度为35的list,每个元素是一个shape=(20, 200)的张量-其实就是这个batch在当前时刻的输出预测值
                                          #表示35时刻的每个样本的
        output=tf.reshape(tf.concat(outputs, 1),[-1,HIDDEN_SIZE])
        # print("output：", output)     #原先的35个大小为(20, 200)的张量在第1维度(列)上合并成后
                                        #大小为(20, 7000)，之后再reshape成shape=(700, 200)
                                        #700行中每35行是batcha中一个样本在35个时刻时的不同状态


        #原来经过cell后得到cell_out，state，两个大小是一样的，而cell_out必须经过最后全联接层变换后
        #才是真正的cell的输出
        weight=tf.get_variable("weight",[HIDDEN_SIZE,VOCAB_SIZE])
        bias=tf.get_variable("bias",[VOCAB_SIZE])
        logits=tf.matmul(output,weight)+bias
        # print("logits:",logits)     #shape=(700, 10000)
        #这里logits的每一行表示每个样本的在当前时刻经过RNN后对于单词的预测，10000表示对单词表中10000个单词的预测概率
        #同样，每35行表示一个样本在35个时刻的预测

        #print(tf.reshape(self.targets,[-1]))   #shape=(700,),targets原是[20,35],现在得到(700,)，同样每35行表示一个样本在35时刻时的
        #输出标签值，这里的顺序与logits是一样的
        #这里的sequence_loss_by_example与sparse_softmax_cross_entropy_with_logits一样，其中logits是(700, 10000)
        #[tf.reshape(self.targets,[-1])是(700,)，同样要将其展开成(700, 10000)计算交叉熵，第三个参数是权重，这里全部设置为1，表示同等权重。
        loss=tf.contrib.legacy_seq2seq.sequence_loss_by_example([logits],[tf.reshape(self.targets,[-1])],
                                                                [tf.ones([batch_size*num_steps],dtype=tf.float32)])


        # print("loss：",loss)   #loss是(700,)表示每个预测的交叉熵，和sparse_softmax_cross_entropy_with_logits
        self.cost=tf.reduce_sum(loss)/batch_size
        # print(self.cost)          #一个实数
        self.final_state=state

        if not is_training:
            return
        trainable_variables=tf.trainable_variables()

        #tf.gradients(ys,xs,...)实现ys对xs求偏导，其中返回值grads长度与xs相同
        grads,_=tf.clip_by_global_norm(tf.gradients(self.cost,trainable_variables),MAX_GARD_NORM)

        # print(len(grads))    #5
        optimizer=tf.train.GradientDescentOptimizer(LEARNING_RATE)
        self.train_op=optimizer.apply_gradients(zip(grads,trainable_variables))   #利用偏导不断梯度下降

def get_a_cell():
   return   tf.nn.rnn_cell.BasicRNNCell(num_units=128)

def run_epoch(session,model,data_queue,train_op,output_log,epoch_size):
    total_cost=0.0
    iters=0
    state=session.run(model.initial_state)
    for step in range(epoch_size):       #每个batch为一个迭代训练集
        feed_dict={}
        x,y=session.run(data_queue)      #每次从reader.ptb_producer的返回值代表的队列中找到一个batch

        # print(x.shape,y.shape)        #每个batch的输入和标签大小都是(20, 35)
        feed_dict[model.input_data]=x   #填充placeholder
        feed_dict[model.targets] = y

        # LSTM可以看做有两个隐状态h和c，对应的隐层就是一个Tuple，每个都是(batch_size, state_size)的形状，state_size就是HIDDEN_SIZE
        #一般的BasicRNNCell单元，只有一个隐藏状态，那就是在隐藏层之后得到的状态(与全联接层计算的那个状态)
        for i,(c,h) in enumerate(model.initial_state):    #如果基本的RNN单元用的是BasicRNNCell，而不是BasicLSTMCell，那么其initial_state就无法循环迭代
            feed_dict[c]=state[i].c
            feed_dict[h]=state[i].h

        cost,state,_=session.run([model.cost,model.final_state,train_op],feed_dict=feed_dict)

        total_cost+=cost
        iters+=model.num_steps

        if output_log and step%100==0:       #计算perplexity值-平均分支数，其越小，其表明NLP模型越好(可以直观理解为在模型生成一句话时预测下一个词时的选择数)
                                             #可选词数越少，我们大致认为模型越准确
            print("经过%d步后，perplexity值为%.3f"%(step,np.exp(total_cost/iters)))

    return np.exp(total_cost/iters)


def main(_):
    train_data,valid_data,test_data,_=reader.ptb_raw_data(DATA_PATH)  #训练数据，验证数据和测试数据都各是一条长序列，各由多个单词元素构成（单词元素以ID表示）

    train_data_len=len(train_data)
    train_batch_len=train_data_len//TRAIN_NUM_STEP   #训练
    train_epoch_size=(train_batch_len-1)//TRAIN_BATCH_SIZE
    # print(train_data_len,train_batch_len,train_epoch_size) #929589 26559 1327 //一共有929589个训练单词，每句有TRAIN_NUM_STEP个单词，最后
                                                             # 有26559个句子，每个batch有TRAIN_BATCH_SIZE=20个句子，就共有，1327个batch

    valid_data_len=len(valid_data)
    valid_batch_len=valid_data_len//EVAL_NUM_STEP
    valid_epoch_size=(valid_batch_len-1)//EVAL_BATCH_SIZE

    test_data_len=len(test_data)
    test_batch_len=test_data_len//EVAL_NUM_STEP
    test_epoch_size=(test_batch_len-1)//EVAL_BATCH_SIZE

    initializer = tf.random_uniform_initializer(-0.05, 0.05)

    with tf.variable_scope("language_model",reuse=None,initializer=initializer):
        train_model=PTBModel(True,TRAIN_BATCH_SIZE,TRAIN_NUM_STEP)

    with tf.variable_scope("language_model",reuse=True,initializer=initializer):
        eval_model=PTBModel(False,EVAL_BATCH_SIZE,EVAL_NUM_STEP)

    train_queue=reader.ptb_producer(train_data,train_model.batch_size,
                                      train_model.num_steps)
    valid_queue = reader.ptb_producer(valid_data, eval_model.batch_size,
                                      eval_model.num_steps)
    test_queue = reader.ptb_producer(test_data, eval_model.batch_size,
                                      eval_model.num_steps)



    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        coord=tf.train.Coordinator()
        threads=tf.train.start_queue_runners(sess=sess,coord=coord)

        for i in range(NUM_EPOCH):
            print("在第%d步迭代中:"%(i+1))
            run_epoch(sess,train_model,train_queue,train_model.train_op,
                      True,train_epoch_size)
            valid_perplexity=run_epoch(sess,eval_model,valid_queue,tf.no_op(),False,
                                       valid_epoch_size)

            print("第%d代：验证集的Perplexity值为:%s"%(i+1,valid_perplexity))

        test_perplexity = run_epoch(sess, eval_model, test_queue,
                                    tf.no_op(), False, test_epoch_size)

        print('最终验证集的Perplexity值为: %.3f' % test_perplexity)

    coord.request_stop()
    coord.join(threads)



if __name__ == '__main__':
    tf.app.run()


