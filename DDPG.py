# +
from Environment import Environment
from Parameter import Parameter
from scipy.io import savemat
import numpy as np
import tensorflow as tf
from tensorflow import keras
import random
import os
import time
import argparse
import sys
import time
import pickle
import pdb

# +
#################### params ###########################
parser = argparse.ArgumentParser(description='Hyper_params')
parser.add_argument('--Info', default='', type=str)  # information added to log dir name

parser.add_argument('--Seed', default=41, type=int)
parser.add_argument('--Units', default=256, type=int)  # hidden units num of NN
parser.add_argument('--Lr', default=0.001, type=float)  # learning rate
parser.add_argument('--omega', default=0.005, type=float)  # used to update target networks
parser.add_argument('--Max_Epsilon', default=1.0, type=float)
parser.add_argument('--Min_Epsilon', default=1.0, type=float)
parser.add_argument('--Epsilon_Decay', default=1.0, type=float)
parser.add_argument('--Batch_Size', default=256, type=int)
parser.add_argument('--Memory_Size', default=1000000, type=int) # buffer size
parser.add_argument('--Start_Size', default=0, type=int)  # random action before start_size
parser.add_argument('--Update_After', default=0, type=int)
parser.add_argument('--Train_Interval', default=1, type=int)
parser.add_argument('--load_weights', default=False, type=bool)
parser.add_argument('--Alg', default='ddpg', type=str)
parser.add_argument('--Gpu_Id', default="0", type=str) # -1 means CPU
parser.add_argument('--N', default=15, type=int)  # number of WDs
parser.add_argument('--T', default=1000000, type=int)  # number of simulated slots
parser.add_argument('--batch_norm', default=True, type=bool)

args = parser.parse_args()
# -

#################### seed ###########################
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = args.Gpu_Id
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
print(tf.config.list_physical_devices())
tf.random.set_seed(args.Seed)
np.random.seed(args.Seed)
random.seed(args.Seed)

#################### log ###########################
# create log file
time_str = time.strftime("%m-%d_%H-%M", time.localtime())
alg = args.Alg
log_dir_name = 'logs/' + time_str + '_' + alg + args.Info + '_n' + \
               str(args.N) + '_seed' + str(args.Seed)
data_dir_name = 'data/' + alg + args.Info + '_n' + str(args.N)
fw = tf.summary.create_file_writer(log_dir_name)  # log file witer

# create dir to save model
if not os.path.exists(log_dir_name + '/models'):
    os.makedirs(log_dir_name + '/models')

# save params to a .txt file
prams_file = open(log_dir_name + '/prams_table.txt', 'w')
prams_file.writelines(f'{i:50} {v}\n' for i, v in args.__dict__.items())
prams_file.close()

###################### env ###############################
param = Parameter(args.N, args.T)
param.lam_init = np.ones(param.N) * 1e4
param.theta = lambda t: 0
param.beta = lambda t: 0.01
env = Environment(param)
if args.load_weights:
    with open('models/v.pickle', 'rb') as f:
        Initial_v = pickle.load(f)
else:
    Initial_v = 0  # initial average reward

###################### others ###############################
W_Initializer = tf.initializers.he_normal(args.Seed)  # NN initializer
Epsilon_Decay_Rate = (args.Min_Epsilon - args.Max_Epsilon) / (args.T) * args.Epsilon_Decay # factor of decay
TENSOR_FLOAT_TYPE = tf.dtypes.float32
TENSOR_INT_TYPE = tf.dtypes.int32

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))

# +
class ReplayBuffer:
    def __init__(self, buffer_capacity = 100000):
        self.buffer_capacity = buffer_capacity
        self.buffer_counter = 0

        # dim(action) = N * 3
        # dim(state) = N * 4
        # dim(pds) = N * 4
        buffer_a_dim = (buffer_capacity, param.N, 3)
        buffer_s_dim = (buffer_capacity, param.N, 4)

        self.s_buffer = np.empty(buffer_s_dim, dtype=np.float32)
        self.a_buffer = np.empty(buffer_a_dim, dtype=np.float32)
        self.r_buffer = np.empty((buffer_capacity,), dtype=np.float32)
        self.next_s_buffer = np.empty(buffer_s_dim, dtype=np.float32)

    def store(self, exp):
        index = self.buffer_counter % self.buffer_capacity

        s, a, r, next_s = exp
        self.s_buffer[index] = s
        self.a_buffer[index] = a
        self.r_buffer[index] = r
        self.next_s_buffer[index] = next_s

        self.buffer_counter += 1

    def sample(self, batch_size):
        sampling_range = min(self.buffer_counter, self.buffer_capacity)
        idx = np.random.randint(0, sampling_range, batch_size)

        batch_s = tf.convert_to_tensor(self.s_buffer[idx])
        batch_a = tf.convert_to_tensor(self.a_buffer[idx])
        batch_r = tf.convert_to_tensor(self.r_buffer[idx])
        batch_next_s = tf.convert_to_tensor(self.next_s_buffer[idx])

        return batch_s, batch_a, batch_r, batch_next_s

class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        # Store x into x_prev
        # Makes next noise dependent on current one
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)
# -

class DPDS:
    def __init__(self, batch_size, memory_size, max_epsilon):

        def build_actor():
            inputs = keras.Input(shape=(param.N, 4))
            x = keras.layers.Flatten()(inputs)
            x = keras.layers.Dense(args.Units, activation='relu', kernel_initializer=W_Initializer)(x)
            if args.batch_norm:
                x = keras.layers.BatchNormalization()(x)
            x = keras.layers.Dense(args.Units, activation='relu', kernel_initializer=W_Initializer)(x)
            if args.batch_norm:
                x = keras.layers.BatchNormalization()(x)
            x = keras.layers.Dense(param.N*3, activation='sigmoid', kernel_initializer=W_Initializer)(x)
            outputs = keras.layers.Reshape((param.N, 3))(x)
            model = keras.Model(inputs=inputs, outputs=outputs)
            return model

        def build_critic():
            state_input = keras.layers.Input(shape=(param.N, 4))
            state_x = keras.layers.Flatten()(state_input)
            #state_x = keras.layers.Dense(args.Critic_Units, activation='relu', kernel_initializer=W_Initializer)(state_x)
            #state_x = keras.layers.Dense(2*args.Critic_Units, activation='relu', kernel_initializer=W_Initializer)(state_x)

            action_input = keras.layers.Input(shape=(param.N, 3))
            action_x = keras.layers.Flatten()(action_input)
            #action_x = keras.layers.Dense(2*args.Critic_Units, activation='relu', kernel_initializer=W_Initializer)(action_x)

            concat = keras.layers.Concatenate()([state_x, action_x])
            x = keras.layers.Dense(args.Units, activation='relu', kernel_initializer=W_Initializer)(concat)
            #if args.batch_norm:
            #    x = keras.layers.BatchNormalization()(x)
            x = keras.layers.Dense(args.Units, activation='relu', kernel_initializer=W_Initializer)(x)
            #if args.batch_norm:
            #    x = keras.layers.BatchNormalization()(x)
            outputs = keras.layers.Dense(1)(x)
            model = keras.Model([state_input, action_input], outputs)
            return model

        if 'ddpg' in alg:
            self.actor = build_actor()
            self.critic = build_critic()
            self.target_actor = build_actor()
            self.target_critic = build_critic()
            
            if args.load_weights:
                self.critic.load_weights("models/critic")
                self.actor.load_weights("models/actor")
                print("load weight")

            self.target_actor.set_weights(self.actor.get_weights())
            self.target_critic.set_weights(self.critic.get_weights())
        else:
            raise NotImplementedError("alg not implemented")

        self.actor_optimizer = tf.optimizers.Adam(args.Lr)
        self.critic_optimizer = tf.optimizers.Adam(args.Lr*2)
        self.epsilon = max_epsilon
        self.batch_size = batch_size
        self.buffer = ReplayBuffer(memory_size)
        self.alg = alg
        self.v = Initial_v  # average reward
        self.target_v = Initial_v
        self.lam = param.lam_init

        # transform some parameters to tensors
        self.f_max = tf.convert_to_tensor(param.f_max)
        self.P_max = tf.convert_to_tensor(param.P_max)
        self.W_max = tf.convert_to_tensor(param.W_max)
        self.E_max = tf.convert_to_tensor(param.E_max)
        self.d_t = tf.convert_to_tensor(param.d_t)
        self.kappa = tf.convert_to_tensor(param.kappa)
        self.sigma2 = tf.convert_to_tensor(param.sigma2)
        self.gamma = tf.convert_to_tensor(param.gamma)

    def random_action(self, s):
        action = np.random.rand(param.N, 3)
        # apply softmax to W so that its sum equals 1
        action[:,2] = softmax(action[:,2])
        return action
    
    def _choose_action(self, s):
        action = self.actor(s[None, :])[0].numpy()
        action[:,2] = softmax(action[:,2])
        return action

    def choose_action(self, s, noise_object, epsilon):
        action = self.actor(s[None, :])[0].numpy()
        noise = noise_object()
        # Adding noise to action
        action = action + epsilon * noise

        # We make sure action is within bounds
        legal_action = np.clip(action, 0, 1)
        legal_action[:,2] = softmax(legal_action[:,2])

        return legal_action

    @tf.function(jit_compile=True)
    def f_k(self, batch_state, batch_action):
        f = batch_action[:, :, 0] * self.f_max
        P = batch_action[:, :, 1] * self.P_max
        W = tf.nn.softmax(batch_action[:, :, 2]) * self.W_max
        d_r = batch_state[:, :, 0]
        a = batch_state[:, :, 1]
        q = batch_state[:, :, 2]
        h = batch_state[:, :, 3]

        d = f * self.d_t / self.kappa + \
                self.d_t * W * (tf.math.log(1+P*h/self.sigma2)/tf.math.log(2.))
        b = ((d > 0) & (d_r > 0) & (d >= d_r))
        b = tf.where(b, 1.0, 0.0)
        pds_q = q - b
        pds_h = h
        pds_d_r = tf.maximum(tf.constant(0, dtype=tf.float32), d_r - d)
        pds_a = a - 5 * b
        
        s = tf.stack([pds_d_r, pds_a, pds_q, pds_h], axis=2)
        return s

    @tf.function(jit_compile=True)
    def cost(self, batch_state, batch_action):
        f = batch_action[:, :, 0] * self.f_max
        P = batch_action[:, :, 1] * self.P_max
        W = tf.nn.softmax(batch_action[:, :, 2]) * self.W_max
        E = self.gamma * f**3 * self.d_t + P * self.d_t
        h = batch_state[:, :, 3]
        d = f * self.d_t / self.kappa + \
                self.d_t * W * (tf.math.log(1+P*h/self.sigma2)/tf.math.log(2.))
        # in expectation, completing a task reduces aoi by 1/p_g
        aoi_per_bit = 1 / param.p_g / ((param.d_lb+param.d_ub)/2)
        cost = tf.reduce_sum(batch_state[:, :, 1] - d*aoi_per_bit + self.lam * tf.math.maximum(0.0, E - self.E_max), axis=1)
        #cost = tf.reduce_sum(self.lam * tf.math.maximum(0.0, E - self.E_max), axis=1)
        #cost = tf.reduce_sum(batch_state[:, :, 1] + self.lam * tf.math.maximum(0.0, E - self.E_max), axis=1)
        #cost = tf.reduce_sum(batch_state[:, :, 1] + self.lam * (E - self.E_max), axis=1)
        return cost

    @tf.function(jit_compile=True)
    def train(self, s, a, r, s_next):
        # update critic network
        with tf.GradientTape() as tape:
            # calculate target y
            target_a_next = self.target_actor(s_next, training=True)
            target_y = self.cost(s,a) + self.target_critic([s_next, target_a_next], training=True) - self.target_v
            critic_value = self.critic([s, a], training=True)
            td = critic_value - target_y
            critic_loss = tf.math.reduce_mean(tf.math.abs(td))

        critic_grad = tape.gradient(critic_loss, self.critic.trainable_variables)
        #critic_grad = [tf.clip_by_norm(grad, 10.0) for grad in critic_grad]
        self.critic_optimizer.apply_gradients( zip(critic_grad, self.critic.trainable_variables) )

        # update actor network
        with tf.GradientTape() as tape:
            actions = self.actor(s, training=True)
            critic_value = self.critic([s, actions], training=True)
            actor_loss = tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, self.actor.trainable_variables)
        #actor_grad = [tf.clip_by_norm(grad, 10.0) for grad in actor_grad]
        self.actor_optimizer.apply_gradients( zip(actor_grad, self.actor.trainable_variables) )

        # td is returned to update self.v
        # we do not update self.v in this function because it leaks the local tensor 'td', which is prohibited by tensorflow
        return (td, critic_loss, actor_loss, actor_grad, critic_grad, critic_value)

    def save_model(self, dir=log_dir_name + '/models'):
        self.actor.save_weights(dir + '/' + self.alg + '_actor')
        self.critic.save_weights(dir + '/' + self.alg + '_critic')
        self.actor.save_weights('models/actor')
        self.critic.save_weights('models/critic')
        with open(dir + '/' + self.alg + '_v.pickle', 'wb') as f:
            pickle.dump(self.v, f)
        with open('models/v.pickle', 'wb') as f:
            pickle.dump(self.v, f)

    
@tf.function(jit_compile=True)
def update_target(target_weights, weights, omega):
    for (a, b) in zip(target_weights, weights):
        a.assign(b * omega + a * (1 - omega))

def train(T):
    agent = DPDS(args.Batch_Size, args.Memory_Size, args.Max_Epsilon)
    print("============" + agent.alg + "============")

    state = env.reset()
    std_dev = 0.01
    ou_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(std_dev) * np.ones(1))
    acc_E = np.zeros(param.N)
    acc_A = np.zeros(param.N)
    timer = 1
    
    acc_interaction_time = 0
    acc_inference_time = 0
    acc_training_time = 0
    
    while timer <= T:
        if timer % 10000 == 0:
            print(timer)

        if timer <= args.Start_Size:
            action = agent.random_action(state)
        else:
            inference_begin = time.time()
            action = agent.choose_action(state, ou_noise, agent.epsilon)
            inference_end = time.time()
            acc_inference_time += inference_end - inference_begin
        interaction_begin = time.time()
        next_state, E = env.step(action)
        interaction_end = time.time()
        acc_interaction_time += interaction_end - interaction_begin
        #cost = np.sum(state[:,1] + agent.lam * (E - param.E_max))
        cost = np.sum(agent.lam * (E - param.E_max))

        agent.buffer.store((state, action, cost, next_state))

        # train
        if timer > args.Update_After and timer % args.Train_Interval == 0:
            training_begin = time.time()
            # sample from buffer
            s, a, r, s_next = agent.buffer.sample(args.Batch_Size)
            td, critic_loss, actor_loss, actor_grad, critic_grad, critic_value = agent.train(s, a, r, s_next)
            #tf.print(actor_grad)
            #tf.print(critic_grad)
            agent.v = agent.v - param.beta(timer) * tf.reduce_mean(td)
            training_end = time.time()
            acc_training_time += training_end - training_begin

            # update target networks
            update_target(agent.target_actor.variables, agent.actor.variables, args.omega)
            update_target(agent.target_critic.variables, agent.critic.variables, args.omega)
            agent.target_v = args.omega * agent.v + (1 - args.omega) * agent.target_v

            # update lambda
            agent.lam = agent.lam + param.theta(timer) * (E - param.E_max)
            agent.lam = np.maximum(0, agent.lam)
            agent.lam = np.minimum(param.lam_max, agent.lam)

            with fw.as_default():
                tf.summary.scalar('critic_loss', critic_loss, step = timer)
                tf.summary.scalar('actor_loss', actor_loss, step = timer)
            
        # epsilon decay
        agent.epsilon = max(Epsilon_Decay_Rate * timer + args.Max_Epsilon, args.Min_Epsilon)

        timer += 1

        # log
        acc_E += E
        acc_A += state[:,1]
        with fw.as_default():
            tf.summary.scalar('cost', cost, step=timer)
            tf.summary.scalar('aoi', np.sum(state[:,1])/args.N, step=timer)
            tf.summary.scalar('average aoi', np.sum(acc_A)/timer/args.N, step=timer)
            tf.summary.scalar('energy', np.sum(E)/args.N, step=timer)
            tf.summary.scalar('v', agent.v, step=timer)
            tf.summary.scalar('epsilon', agent.epsilon, step=timer)
            tf.summary.scalar('lambda', np.sum(agent.lam)/args.N, step=timer)
            tf.summary.scalar('average energy', np.sum(acc_E)/timer/args.N, step=timer)

        state = next_state
    
    agent.save_model()
    print("Average interaction time:", acc_interaction_time / args.T)
    print("Average inference time:", acc_inference_time / (args.T - args.Start_Size))
    print("Average training time:", acc_training_time / (args.T - args.Update_After))
    print("Average number of tasks:", env.nTask / timer /args.N)

    data = {'N': param.N, 'E': env.E_stat, 'A': env.A_stat}
    savemat(data_dir_name + '.mat', data)

if __name__ == "__main__":
    train(args.T)

print(tf.__version__)


