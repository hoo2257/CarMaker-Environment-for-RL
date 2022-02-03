import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import os

class ReplayBuffer:
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape))
        self.new_state_memory = np.zeros((self.mem_size, *input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, states_, dones

class CriticNetwork(keras.Model):
    def __init__(self, name, chkpt_dir="model_cm_ftp_v0.1/td3"):
        super(CriticNetwork, self).__init__()

        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+"_td3")

        self.fc1 = Dense(128, activation='relu')
        self.fc2 = Dense(64, activation='relu')
        self.fc3 = Dense(32, activation='relu')
        self.fc4 = Dense(16, activation='relu')
        self.fc5 = Dense(8, activation='relu')

        self.q = Dense(1, activation=None)

    def call(self, state, action):
        q1_action_value = self.fc1(tf.concat([state, action], axis=1))
        q1_action_value = self.fc2(q1_action_value)
        q1_action_value = self.fc3(q1_action_value)
        q1_action_value = self.fc4(q1_action_value)
        q1_action_value = self.fc5(q1_action_value)


        q = self.q(q1_action_value)

        return q

class ActorNetwork(keras.Model):
    def __init__(self, n_actions, name, chkpt_dir="model_cm_ftp_v0.1/td3"):
        super(ActorNetwork, self).__init__()
        self.n_actions = n_actions
        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+"_td3")

        self.fc1 = Dense(128, activation='relu')
        self.fc2 = Dense(64, activation='relu')
        self.fc3 = Dense(32, activation='relu')
        self.fc4 = Dense(16, activation='relu')
        self.fc5 = Dense(8, activation='relu')
        self.mu = Dense(self.n_actions, activation="tanh")

    def call(self, state):
        prob = self.fc1(state)
        prob = self.fc2(prob)
        prob = self.fc3(prob)
        prob = self.fc4(prob)
        prob = self.fc5(prob)
        prob = self.fc6(prob)
        prob = self.fc7(prob)
        prob = self.fc8(prob)

        mu = 5*self.mu(prob)

        return mu

class Agent:
    def __init__(self, alpha, beta, input_dims, tau, model_chkpt_dir, gamma = 0.99,
                 update_actor_interval=5, warmup=4,
                 n_actions = 1, max_size=3000000, batch_size=128, noise = 3.0):
        self.gamma = gamma
        self.tau = tau
        self.max_action = 5
        self.min_action = -5
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.learn_step_cntr = 0
        self.time_step=0
        self.warmup = warmup
        self.n_actions = n_actions
        self.update_actor_iter = update_actor_interval

        self.actor = ActorNetwork(n_actions=n_actions, name="actor", chkpt_dir=model_chkpt_dir)
        self.critic_1 = CriticNetwork(name="critic_1", chkpt_dir=model_chkpt_dir)
        self.critic_2 = CriticNetwork(name="critic_2", chkpt_dir=model_chkpt_dir)

        self.target_actor = ActorNetwork(n_actions=n_actions, name="target_actor")
        self.target_critic_1 = CriticNetwork(name="target_critic_1")
        self.target_critic_2 = CriticNetwork(name="target_critic_2")
        self.actor.compile(optimizer=Adam(learning_rate=alpha), loss="mean")
        self.critic_1.compile(optimizer=Adam(learning_rate=beta),
                              loss="mean_squared_error")
        self.critic_2.compile(optimizer=Adam(learning_rate=beta),
                              loss="mean_squared_error")

        self.target_actor.compile(optimizer=Adam(learning_rate=alpha),
                                  loss="mean")
        self.target_critic_1.compile(optimizer=Adam(learning_rate=beta),
                                     loss="mean_squared_error")
        self.target_critic_2.compile(optimizer=Adam(learning_rate=beta),
                                     loss="mean_squared_error")
        self.noise = noise
        self.update_network_parameters(tau=1)

    def choose_action(self, observation, episode):
        if episode < self.warmup:  # If episode is smaller than self.warmup
            mu = np.random.normal(scale=self.noise, size=(self.n_actions,))  # Select Random Action

        else:
            state = tf.convert_to_tensor([observation], dtype=tf.float32) # Else feed state to actor to get action
            mu = self.actor(state)[0]

            # print("mu ", mu)

        mu_prime = mu + np.random.normal(scale=0.01) # Add Noise to the action for exploration
        mu_prime = tf.clip_by_value(mu_prime, self.min_action, self.max_action)

        return mu_prime

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def learn(self):
        if self.memory.mem_cntr < self.batch_size: # if replay buffer size is smaller than batch size, break
            return

        states, actions, rewards, new_states, dones = self.memory.sample_buffer(self.batch_size)
        states = tf.convert_to_tensor(states, dtype=tf.float32) # Current State
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        states_ = tf.convert_to_tensor(new_states, dtype=tf.float32) # Next State

        with tf.GradientTape(persistent=True) as tape: # Learning Process for Critic Networks
            target_actions = self.target_actor(states_) # Feed Next State to target Actor
            target_actions = target_actions + tf.clip_by_value(np.random.normal(scale=0.2), -0.5, 0.5)
            target_actions = tf.clip_by_value(target_actions, self.min_action, self.max_action)

            q1_ = self.target_critic_1(states_, target_actions)
            q2_ = self.target_critic_2(states_, target_actions)

            # shape is [batch_size, 1] want to collapse to [batch_size]
            q1_ = tf.squeeze(q1_, 1)
            q2_ = tf.squeeze(q2_, 1)

            q1 = tf.squeeze(self.critic_1(states, actions), 1)
            q2 = tf.squeeze(self.critic_2(states, actions), 1)

            critic_value_ = tf.math.minimum(q1_, q2_)

            target = rewards + self.gamma*critic_value_*(1-dones)

            critic_1_loss = keras.losses.MSE(target, q1)
            critic_2_loss = keras.losses.MSE(target, q2)

            # print("Critic1Loss")
            # print(critic_1_loss)
            # print("Critic1Loss")
            # print(critic_2_loss)

        critic_1_gradient = tape.gradient(critic_1_loss,
                                        self.critic_1.trainable_variables)
        critic_2_gradient = tape.gradient(critic_2_loss,
                                        self.critic_2.trainable_variables)

        self.critic_1.optimizer.apply_gradients(zip(critic_1_gradient, self.critic_1.trainable_variables))
        self.critic_2.optimizer.apply_gradients(zip(critic_2_gradient, self.critic_2.trainable_variables))

        self.learn_step_cntr += 1

        if self.learn_step_cntr % self.update_actor_iter !=0:
            return

        with tf.GradientTape() as tape:
            new_actions = self.actor(states)
            critic_1_value = self.critic_1(states, new_actions)
            actor_loss = -tf.math.reduce_mean(critic_1_value)

        actor_gradient = tape.gradient(actor_loss, self.actor.trainable_variables)
        # print(actor_gradient)
        self.actor.optimizer.apply_gradients(zip(actor_gradient, self.actor.trainable_variables))

        self.update_network_parameters()

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        weights = []
        targets = self.target_actor.weights
        for i, weight in enumerate(self.actor.weights):
            weights.append(weight*tau + targets[i]*(1-tau))
        self.target_actor.set_weights(weights)

        weights = []
        targets = self.target_critic_1.weights
        for i, weight in enumerate(self.critic_1.weights):
            weights.append(weight * tau + targets[i] * (1 - tau))
        self.target_critic_1.set_weights(weights)

        weights = []
        targets = self.target_critic_2.weights
        for i, weight in enumerate(self.critic_2.weights):
            weights.append(weight * tau + targets[i] * (1 - tau))
        self.target_critic_2.set_weights(weights)

    def save_models(self):
        print("... saving models ...")
        self.actor.save_weights(self.actor.checkpoint_file)
        self.critic_1.save_weights(self.critic_1.checkpoint_file)
        self.critic_2.save_weights(self.critic_2.checkpoint_file)
        self.target_actor.save_weights(self.target_actor.checkpoint_file)
        self.target_critic_1.save_weights(self.target_critic_1.checkpoint_file)
        self.target_critic_2.save_weights(self.target_critic_2.checkpoint_file)

    def load_models(self):
        print("... loading models ...")
        self.actor.load_weights(self.actor.checkpoint_file)
        self.critic_1.load_weights(self.critic_1.checkpoint_file)
        self.critic_2.load_weights(self.critic_2.checkpoint_file)
        self.target_actor.load_weights(self.target_actor.checkpoint_file)
        self.target_critic_1.load_weights(self.target_critic_1.checkpoint_file)
        self.target_critic_2.load_weights(self.target_critic_2.checkpoint_file)

    def load_actor(self):
        print("... loading Trained Actor ...")
        self.actor.load_weights(self.actor.checkpoint_file)






