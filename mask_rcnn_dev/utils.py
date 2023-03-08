from matplotlib import pyplot as plt

def show_maximized():
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.show()