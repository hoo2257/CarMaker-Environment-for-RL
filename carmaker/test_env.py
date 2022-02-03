from CarMakerEnv import *
from td3_tf2 import *
from utils import gpu_setting
import pandas as pd
import numpy as np
import time

if __name__ == "__main__":
    gpu_setting(3)
    n_episode = 1000
    score_avg = 0
    score_history = []

    best_score = -5000000000
    past_score = -200000
    drive_cycle = "ftp"
    model_name = "model_cm_ftp_v0.1/td3"
    window = 15
    state_size = 4*window
    target_data = pd.read_csv(f"target_data/sim/target_{drive_cycle}_real.csv", names=["target"])
    target_data = list(target_data["target"])

    cnt = 0
    scores, episode, logger_score = [], [], []

    agent = Agent(alpha=0.0001, beta=0.0002, input_dims=(state_size,),
                  tau=0.005, batch_size=64, n_actions=1
                  ,model_chkpt_dir=model_name)


    for n in range(n_episode):
        env = CMEnv(window)
        time.sleep(5)
        tg_queue, fut_tg_list,\
        pvel_queue, fvel_queue,state, score = env.reset()
        state = state[0]

        logger_reward = []

        env.signal = 0
        action = agent.choose_action(state, n)
        env.get_signal(action)

        env.start()
        env.send_signal()

        for i, value in enumerate(target_data):
            velocity, sim_time, consump, exp_vel, _ = env.recv_data()

            print(sim_time, " /" , i/10)

            error = value - velocity

            tg_queue.pop(0)
            tg_queue.append(value)
            fut_tg_list = target_data[i:i+env.window]
            # sig_queue.pop(0)
            # sig_queue.append(env.signal)
            pvel_queue.pop(0)
            pvel_queue.append(velocity)
            fvel_queue.pop(0)
            fvel_queue.append(exp_vel)

            reward, next_state = env.step(tg_queue, fut_tg_list,
                                          pvel_queue, fvel_queue, consump, error)

            next_state = next_state[0]

            if i%2==0:
                done = 0
                agent.remember(state, action, reward, next_state, done)
            agent.learn()

            score += reward

            logger_reward.append(reward)

            state = next_state
            action = agent.choose_action(state, n)
            signal = env.get_signal(action)
            env.send_signal()
            slp = (i / 10) - sim_time

            if slp > 0:
                time.sleep(slp)
            else:
                pass

            if i == 3500:
                break

        env.finish()
        cnt+=1
        print("[INFO] Drive Cycle Complete")
        score_history.append(score)
        score_avg = np.mean(score_history[-20:])

        if score_avg > best_score:
            best_score = score_avg
            agent.save_models()

        logger_score.append(score)
        scores.append(score_avg)
        episode.append(cnt)

        log = "[INFO] episode: {:5d} | ".format(n)
        log += "score: {:4.1f} | ".format(score)
        log += "score max: {:4.1f} | ".format(best_score)
        log += "score avg: {:4.1f} | ".format(score_avg)

        print(log)

        avg_plot = plt.figure()
        plt.plot(episode, scores, 'b', label='average_score')
        plt.plot(episode, logger_score, 'g--', alpha=0.5, label='score')
        plt.xlabel('episode')
        plt.ylabel('score')

        plt.legend()
        plt.savefig("TD3_result/save_graph/graph.png")
        plt.close(avg_plot)

        try:
            os.mkdir("TD3_result/save_graph/reward")
        except:
            pass

        # if n % 100 == 0:
        #     plot_Status(logger_vel, logger_tg,logger_reward,n)

        log_train = pd.DataFrame([episode, logger_score, scores], index=["episode", "score", "score_avg"])
        log_train = log_train.transpose()
        log_train.to_csv('train_results/log_train_{}.csv'.format(model_name.split("/")[0]))
        time.sleep(5)









# open_CarMaker()
# env = CMEnv()
# time.sleep(1)
# env.cm.sim_start()
# env.send_data(0.)
# _, _,_, status = env.recv_data()
# print(status)
#
#
#
#
# for i, value in enumerate(signal):
#     env.send_data(value)
#     velocity, sim_time,consump, status = env.recv_data()
#
#     # print(status)
#     print(velocity,consump, sim_time, i/10)
#     # print(time.time()-st, sim_time)
#
#     slp = (i/10) - sim_time
#
#     if slp > 0:
#         time.sleep(slp)
#     else:
#         pass
#
#     if status == 0.0:
#         break
#
# env.cm.sim_stop()
