import matplotlib.pyplot as plt
import tensorflow as tf
import os

def open_CarMaker():
    os.system('cmd /k "C:\IPG\carmaker\win64-10.2\bin\CM.exe -cmdport 16660"')

def scale(value, min_value, max_value, resolution = 5):
    s_data = (value - min_value) / (max_value - min_value)
    s_data = round(s_data,resolution)
    return s_data

def list_min_max(data, min, max, resolution= 5):
    s_data = []
    for value in data:
        scaled_data = scale(value, min, max)
        s_data.append(round(scaled_data, resolution))
    return s_data

def plot_Status(vel,tg,reward,cnt):
    # reward_seq_ = []
    # for n in reward:
    #     for i in range(10):
    #         reward_seq_.append(n)
    reward_plot = plt.figure(figsize=(15,10))
    plt.plot(vel, label="Velocity")
    plt.plot(tg, label="Target")
    plt.plot(reward, label="Reward")
    # plt.plot(zero_crossing, label="Zero Crossing")
    # plt.plot(cur_rmse, label="Cur Rmse")
    # plt.plot(err_mean, label="Error Mean")
    # plt.plot(err_std, label="Err Std")
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Reward")
    plt.savefig("TD3_result/save_graph/reward/reward{}.png".format(cnt))
    plt.close(reward_plot)

def gpu_setting(memory):
    ################### Limit GPU Memory ###################
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print("########################################")
    print('{} GPU(s) is(are) available'.format(len(gpus)))
    print("########################################")

    # set the only one GPU and memort limit
    memory_limit = memory * 1024

    if gpus:
        try:
            tf.config.experimental.set_virtual_device_configuration(gpus[0], [
                tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit)])
            print("Use only one GPU{} limited {}MB memory".format(gpus[0], memory_limit))
        except RuntimeError as e:
            print(e)

    else:
        print('GPU is not available')
    ##########################################################