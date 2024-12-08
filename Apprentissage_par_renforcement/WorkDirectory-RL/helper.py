import matplotlib.pyplot as plt
from IPython import display

plt.ion()

def plot(rewards, steps):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title("Training...")
    plt.xlabel("Number ")
    plt.ylabel("Rewards")
    plt.plot(rewards)
    plt.plot(steps)
    plt.ylim(ymin=-50, ymax=50)
    plt.text(len(rewards) - 1, rewards[-1], str(rewards[-1]))
    plt.text(len(steps) - 1, steps[-1], str(steps[-1]))
    plt.show(block=False)
    #plt.pause(.1)