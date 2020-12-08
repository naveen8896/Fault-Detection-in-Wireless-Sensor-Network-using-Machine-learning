import random
import socket
import time
import pickle
#import struct
s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
s.bind(("127.0.0.1",9996))
count1=0
host="127.0.0.1" #multi cast ip

while(1):
    var1 = random.uniform(49,53)
    var2 = random.uniform(31,32)
    var = [6,var1,var2]
    print(var)
    data=pickle.dumps(var)
    s.sendto(data,(host,9889))
    time.sleep(1)
        #print("msg send no of times",msg)
s.close()



