import matplotlib.pyplot as plt



fontsize=44
ticksize=36


def topk_distribution():
    percents = [93.1, 89.3, 86.1, 75.0]
    x = range(len(percents))
    fig, ax = plt.subplots()

    plt.bar(x, percents,
            tick_label=["100%", "25%", "10%", "1%", ],
            fc='dimgray')


    plt.xlabel('Sampling Rate', fontsize=fontsize)
    plt.ylabel('Top5% Percentage', fontsize=fontsize)

    for i in range(len(percents)):
        plt.text(x = i - 0.12 , y = percents[i] + 0.2, s = str(percents[i]), size = 24, color='blue')


    plt.xticks(fontsize=ticksize - 8)
    plt.yticks(fontsize=ticksize)

    plt.show()


if __name__ == '__main__':
    topk_distribution()
