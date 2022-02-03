from pycarmaker import CarMaker, Quantity
from utils import *
import time
import numpy as np
import os

#
class CMEnv():

    def __init__(self, window):

        IP_ADDRESS = "localhost"
        PORT = 16660
        cm_path = "C:\\IPG\\carmaker\\win64-10.2\\bin"
        cur_path = "E:\\과제\\2021\\현대차 RobotDriver\\carmaker"
        os.chdir(cm_path)
        os.system("CM.exe -cmdport 16660")
        os.chdir(cur_path)

        self.window = window
        self.alpha = 0.9
        self.signal = 0
        self.velocity_ = 0
        self.state_size = 4*self.window

        self.cm = CarMaker(IP_ADDRESS, PORT)
        self.cm.connect()
        self.cm.load_test_run()
        self.cm.sim_start()
        time.sleep(0.5)
        self.cm.sim_stop()
        print("[INFO] CarMaker Initialized")
        time.sleep(1)

        # Subscribing to CarMaker Vehicle Model Parameters >> IDs are listed in ReferenceManual.pdf
        self.sim_status = Quantity("SimStatus", Quantity.INT, True)
        self.sim_time = Quantity("Time", Quantity.FLOAT)
        self.velocity = Quantity("Car.v", Quantity.FLOAT)
        self.elec_cosump = Quantity("PT.Control.Consump.Elec.Act", Quantity.FLOAT)
        self.cm.subscribe(self.sim_status)
        self.cm.subscribe(self.velocity)
        self.cm.subscribe(self.sim_time)
        self.cm.subscribe(self.elec_cosump)

        # Vehicle Parameters created in Python Env
        self.accel = Quantity("DM.Gas", Quantity.FLOAT)
        self.brake = Quantity("DM.Brake", Quantity.FLOAT)

        self.cm.read()
        self.cm.read()


    def start(self):
        """
        Start the TestRun
        """
        self.cm.sim_start()

    def finish(self):
        """
        End the Test Run
        """
        self.cm.sim_stop()
        time.sleep(5)
        os.system('taskkill /IM "' + "HIL.exe" + '" /F')


    def recv_data(self):
        """
        Receive data from CarMaker
        The data must be subscribed at the initialize process
        :return: Received value
        """
        self.cm.read()
        vel = self.velocity.data*3.6
        time = self.sim_time.data
        consump = self.elec_cosump.data*0.1
        status = self.sim_status.data

        acc = (vel - self.velocity_)*0.2778/0.1
        exp_vel = vel + acc*1*3.6

        return vel, time, consump, exp_vel, status

    def reset(self):
        """
        State 및 Score 초기화
        :return: 초기 값
        """
        init_tg = [0] * self.window
        init_fut_tg = [0] * self.window
        # init_signal = [0] * self.window
        init_pvel = [0] * self.window
        init_fvel = [0] * self.window

        init_state = init_tg + init_fut_tg + init_pvel + init_fvel
        self.state_size = len(init_state)
        init_state = np.reshape(init_state, [1, self.state_size])

        score = 0

        return init_tg, init_fut_tg, init_pvel, init_fvel, init_state, score


    def step(self, tg_list, fut_list, sig_list, pvel_list, fvel_list, consump, error):
        """

        :return: REWARD, NEXT STATE
        """
        s_tg_list = list_min_max(tg_list, -10, 140)
        s_fut_list = list_min_max(fut_list, -10, 140)
        # s_sig_list = list_min_max(sig_list, -40, 80)
        s_pvel_list = list_min_max(pvel_list, -10, 140)
        s_fvel_list = list_min_max(fvel_list, -10.,140)

        next_state = s_tg_list + s_fut_list + s_pvel_list + s_fvel_list
        next_state = np.reshape(next_state, [1, self.state_size])
        if consump < 0:
            consump = 0
        else:
            pass

        bonus = 1.0 if abs(error) < 1.0 else 0

        reward = self.alpha*(-abs(error) + bonus) - (1-self.alpha)*(consump)

        return reward, next_state

    def get_signal(self, dx, min_sig=-40, max_sig=80, inference=False):

        if inference:
            dx = dx[0]
        else:
            dx = dx.numpy()[0]

        self.signal = self.signal + dx
        self.signal = np.clip(self.signal, min_sig, max_sig)

    def send_signal(self):
        """
        Send APS, BPS to CarMaker
        :param signal: APS or BPS Signal
        """
        self.signal = round(self.signal, 2)
        # print(self.signal)
        if self.signal > 0:
            self.cm.DVA_release()
            self.cm.DVA_write(self.accel, self.signal * 0.01)

        elif self.signal == 0:
            self.cm.DVA_release()
            self.cm.DVA_write(self.accel, self.signal * 0.01)
            self.cm.DVA_write(self.brake, self.signal * 0.01)

        else:
            self.cm.DVA_release()
            self.cm.DVA_write(self.brake, round(self.signal * -0.01, 3))



















