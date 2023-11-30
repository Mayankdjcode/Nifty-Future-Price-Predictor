import Live_Stock_Prediction_3
import time
import threading

list_of_threads = []
while(1):
    t = threading.Thread(target=Live_Stock_Prediction_3.runModel)
    list_of_threads.append(t)
    t.start()
    print("Waiting 3 minutes")
    time.sleep(180)