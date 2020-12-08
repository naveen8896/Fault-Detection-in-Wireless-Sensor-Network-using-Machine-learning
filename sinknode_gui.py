import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tkinter as tk
from tkinter import *
import threading
import time
from PIL import ImageTk,Image
root = tk.Tk()
root.wm_title("Sink GUI")

dataset = pd.read_csv('Book1.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
acc = accuracy_score(y_test, y_pred)
print("The accuracy is:",acc*100)

from matplotlib.colors import ListedColormap
X_set, y_set = sc.inverse_transform(X_train), y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 0.2, stop = X_set[:, 0].max() + 0.2, step = 0.1),
                     np.arange(start = X_set[:, 1].min() - 0.2, stop = X_set[:, 1].max() + 0.2, step = 0.1))
plt.contourf(X1, X2, classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('SVM (Training set)')
plt.xlabel('Humidity')
plt.ylabel('Temperature')
plt.legend()
plt.show()

from matplotlib.colors import ListedColormap
X_set, y_set = sc.inverse_transform(X_test), y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 0.2, stop = X_set[:, 0].max() + 0.2, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 0.2, stop = X_set[:, 1].max() + 0.2, step = 0.01))
plt.contourf(X1, X2, classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('SVM (Test set)')
plt.xlabel('Humidity')
plt.ylabel('Temperature')
plt.legend()
plt.show()

root.geometry("1200x1200")
l2=Label(root,text="NODE1",bg="light green")
l2.grid(row=3,column=21,padx=0,pady=0)

l3=Label(root,text="NODE2",bg="light green")
l3.grid(row=9,column=17,padx=0,pady=0)

l4=Label(root,text="NODE3",bg="light green")
l4.grid(row=29,column=21,padx=0,pady=0)

l5=Label(root,text="NODE4",bg="light green")
l5.grid(row=3,column=13,padx=0,pady=0)

l6=Label(root,text="NODE5",bg="light green")
l6.grid(row=29,column=13,padx=0,pady=0)

l7=Label(root,text="NODE6",bg="light green")
l7.grid(row=6,column=7,padx=0,pady=0)

l8=Label(root,text="NODE7",bg="light green")
l8.grid(row=13,column=7,padx=0,pady=0)

l9=Label(root,text="NODE8",bg="light green")
l9.grid(row=3,column=1,padx=0,pady=0)

l0=Label(root,text="NODE9",bg="light green")
l0.grid(row=9,column=1,padx=0,pady=0)

l1=Label(root,text="NODE10",bg="light green")
l1.grid(row=29,column=1,padx=0,pady=0)

l10=Label(root,text="SINKNODE",bg="light green")
l10.grid(row=9,column=25,padx=0,pady=0)

my1=ImageTk.PhotoImage(Image.open("12.png"))   
mylabel=Label(root,image=my1)
mylabel.grid(row=2,column=21,padx=0,pady=0, ipadx=0, sticky=N)


my2=ImageTk.PhotoImage(Image.open("12.png"))
mylabel2=Label(root,image=my2)
mylabel2.grid(row=8,column=17,padx=0,pady=0, ipadx=0, sticky=N)

my3=ImageTk.PhotoImage(Image.open("12.png"))
mylabel3=Label(root,image=my3)
mylabel3.grid(row=28,column=21,padx=0,pady=0, ipadx=0, sticky=N)

my4=ImageTk.PhotoImage(Image.open("12.png"))
mylabel4=Label(root,image=my4)
mylabel4.grid(row=2,column=13,padx=0,pady=0, ipadx=0, sticky=N)

my5=ImageTk.PhotoImage(Image.open("12.png"))
mylabel5=Label(root,image=my5)
mylabel5.grid(row=28,column=13,padx=0,pady=0, ipadx=0, sticky=N)

my6=ImageTk.PhotoImage(Image.open("12.png"))
mylabel6=Label(root,image=my6)
mylabel6.grid(row=4,column=7,padx=0,pady=0, ipadx=0, sticky=N)

my7=ImageTk.PhotoImage(Image.open("12.png"))
mylabel7=Label(root,image=my7)
mylabel7.grid(row=12,column=7,padx=0,pady=0, ipadx=0, sticky=N)

my8=ImageTk.PhotoImage(Image.open("12.png"))
mylabel8=Label(root,image=my8)
mylabel8.grid(row=2,column=1,padx=0,pady=0, ipadx=0, sticky=N)

my9=ImageTk.PhotoImage(Image.open("12.png"))
mylabel9=Label(root,image=my9)
mylabel9.grid(row=8,column=1,padx=0,pady=0, ipadx=0, sticky=N)

my10=ImageTk.PhotoImage(Image.open("12.png"))
mylabel10=Label(root,image=my10)
mylabel10.grid(row=28,column=1,padx=0,pady=0, ipadx=0, sticky=N)

my11=ImageTk.PhotoImage(Image.open("14.png"))
mylabel11=Label(root,image=my11)
mylabel11.grid(row=8,column=25,padx=0,pady=0, ipadx=0, sticky=N)

close_thread=False


def sock():
  List=[]
  import socket
  import pickle
    
  s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
  print("sink node created")
  s.bind(("127.0.0.1",9889))
  print("sink node start recieving data")
  
  count1=0
  count2=0
  count3=0
  count4=0
  count5=0
  count6=0
  count7=0
  count8=0
  count9=0
  count10=0
  
  while(1):
    (cdata,addr)=s.recvfrom(2048)
    mdata = pickle.loads(cdata)
    List.append(mdata)
    print("data rx",mdata,addr)
    x=classifier.predict(sc.transform([[mdata[1],mdata[2]]]))
    #print(x[0])
    
    if(mdata[0] == 1):
        if(x[0] == 0):
            count1 = count1 + 1
            if(count1 >= 10):
                print("node1 is the one which is providing faulty data")
                my21=ImageTk.PhotoImage(Image.open("13.png"))   
                mylabel21=Label(root,image=my21)
                mylabel21.grid(row=2,column=21,padx=0,pady=0, ipadx=0, sticky=N)
        else:
            count1 = 0;
                
    elif(mdata[0] == 2):
        if(x[0] == 0):
            count2 = count2 + 1
            if(count2 >= 10):
                print("node2 is the one which is providing faulty data")
                my22=ImageTk.PhotoImage(Image.open("13.png"))
                mylabel22=Label(root,image=my2)
                mylabel22.grid(row=8,column=17,padx=0,pady=0, ipadx=0, sticky=N)
        else:
            count2 = 0;
                
    elif(mdata[0] == 3):
        if(x[0] == 0):
            count3 = count3 + 1
            if(count3 >= 10):
                print("node3 is the one which is providing faulty data")
                my23=ImageTk.PhotoImage(Image.open("13.png"))
                mylabel23=Label(root,image=my23)
                mylabel23.grid(row=28,column=21,padx=0,pady=0, ipadx=0, sticky=N)
        else:
            count3 = 0;
            
    elif(mdata[0] == 4):
        if(x[0] == 0):
            count4 = count4 + 1
            if(count4 >= 10):
                print("node4 is the one which is providing faulty data")
                my24=ImageTk.PhotoImage(Image.open("13.png"))
                mylabel24=Label(root,image=my4)
                mylabel24.grid(row=2,column=13,padx=0,pady=0, ipadx=0, sticky=N)
        else:
            count4 = 0;
            
    elif(mdata[0] == 5):
        if(x[0] == 0):
            count5 = count5 + 1
            if(count5 >= 10):
                print("node5 is the one which is providing faulty data")
                my25=ImageTk.PhotoImage(Image.open("13.png"))
                mylabel25=Label(root,image=my25)
                mylabel25.grid(row=28,column=13,padx=0,pady=0, ipadx=0, sticky=N)
        else:
            count5 = 0;
            
    elif(mdata[0] == 6):
        if(x[0] == 0):
            count6 = count6 + 1
            if(count6 >= 10):
                print("node6 is the one which is providing faulty data")
                my26=ImageTk.PhotoImage(Image.open("13.png"))
                mylabel26=Label(root,image=my26)
                mylabel26.grid(row=4,column=7,padx=0,pady=0, ipadx=0, sticky=N)
        else:
            count6 = 0;
            
    elif(mdata[0] == 7):
        if(x[0] == 0):
            count7 = count7 + 1
            if(count7 >= 10):
                print("node7 is the one which is providing faulty data")
                my27=ImageTk.PhotoImage(Image.open("13.png"))
                mylabel27=Label(root,image=my27)
                mylabel27.grid(row=12,column=7,padx=0,pady=0, ipadx=0, sticky=N)
        else:
            count7 = 0;
            
    elif(mdata[0] == 8):
        if(x[0] == 0):
            count8 = count8 + 1
            if(count8 >= 10):
                print("node8 is the one which is providing faulty data")
                my28=ImageTk.PhotoImage(Image.open("13.png"))
                mylabel28=Label(root,image=my28)
                mylabel28.grid(row=2,column=1,padx=0,pady=0, ipadx=0, sticky=N)
        else:
            count8 = 0;
            
    elif(mdata[0] == 9):
        if(x[0] == 0):
            count9 = count9 + 1
            if(count9 >= 10):
                print("node9 is the one which is providing faulty data")
                my29=ImageTk.PhotoImage(Image.open("13.png"))
                mylabel29=Label(root,image=my29)
                mylabel29.grid(row=8,column=1,padx=0,pady=0, ipadx=0, sticky=N)
        else:
            count9 = 0;
            
    elif(mdata[0] == 10):
        if(x[0] == 0):
            count10 = count10 + 1
            if(count10 >= 10):
                print("node10 is the one which is providing faulty data")
                my30=ImageTk.PhotoImage(Image.open("13.png"))
                mylabel30=Label(root,image=my30)
                mylabel30.grid(row=28,column=1,padx=0,pady=0, ipadx=0, sticky=N)  
        else:
            count10 = 0; 

    z=bytearray(b"\x05\x08\xff\xff\xff\xff\xff")
    s.sendto(z,addr)       
  s.close()
    
thread=threading.Thread(target=sock)
thread.start() 

#print("out of loop",x)
#root.after(1000,sock)
root.mainloop()




