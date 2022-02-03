from pycarmaker import CarMaker, Quantity
import time

# 1 - Initialize pyCarMaker
IP_ADDRESS = "localhost"
PORT = 16660
cm = CarMaker(IP_ADDRESS, PORT)

# 2 - Connect to CarMaker
cm.connect()
time.sleep(1)


# 3 - Create a Quantity
sim_status = Quantity("SimStatus", Quantity.INT, True)
sim_time = Quantity("Time", Quantity.FLOAT)
velocity = Quantity("Car.v", Quantity.FLOAT)
accel = Quantity("DM.Gas", Quantity.FLOAT)
brake = Quantity("DM.Brake", Quantity.FLOAT)
cm.subscribe(sim_status)
cm.subscribe(velocity)
cm.subscribe(sim_time)

cm.read()
cm.read()
print(sim_status.data)
print("WAITING...")

for i in range(3):
    cnt = 0
    print("EPISDE ",i)
    cm.sim_start()
    while True:
        cm.read()
        print(velocity.data, sim_time.data)
        cm.DVA_write(accel, 0.5)
        time.sleep(0.5)


        cm.DVA_write(accel, 0)
        cm.DVA_release()

        cm.DVA_write(brake, 0.5)
        time.sleep(0.5)
        cm.DVA_write(brake, 0)
        cm.DVA_release()

        if cnt > 5:
            cm.sim_stop()
            print("SIM STATS ", sim_status.data)
            break

        cnt+=1

