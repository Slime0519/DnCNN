import matplotlib.pyplot as plt
import numpy as np

def getlength(array):
    for i,value in enumerate(array[0]):
        print(value)
        if value == 0:
            return i
    return 50

# draw graph
result_array = np.load('./psnr_test/psnr_testresult.npy')
length = getlength(result_array)
x = range(length)
y_set12 = result_array[0][:length]
y_set68 = result_array[1][:length]
plot_set12 = plt.plot(x,y_set12,label = 'set12')
plot_set68 = plt.plot(x,y_set68,label = 'set68')
plt.legend()
plt.show()

