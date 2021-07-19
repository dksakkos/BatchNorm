from matplotlib import pyplot as plt

def create_plots(histories, key, title, names=('None', 'Custom', 'Default')):
    for h in histories:
        plt.plot(h.history[key])
    plt.title(title)
    plt.ylabel(key)
    plt.xlabel('epoch')
    plt.legend(names, loc='lower right')
    plt.savefig(key + '.png', dpi=500)
    plt.close()
