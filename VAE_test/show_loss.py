import matplotlib.pyplot as plt
import pickle

# with open('BCE_LossList', 'rb') as f:
with open('MSE_LossList', 'rb') as f:
    loss_list = pickle.load(f)
    f.close()

x = list(range(len(loss_list)))
y = loss_list

plt.plot(x, y)
plt.xlabel('step')
plt.ylabel('loss')

plt.show()
