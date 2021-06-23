import numpy as np
from pylab import rcParams
import matplotlib.pyplot as plt


def get_mean_reward(reward_lst, batch_size):
    mean_rew=list()
    for r in range(len(reward_lst)):
        mean_rew.append(sum(reward_lst[:r+1]) * 1.0 / ((r+1)*batch_size))
    return mean_rew

def get_cum_reward(reward_lst):
    cnt = 0
    new = []
    for i in reward_lst:
        cnt += i
        new.append(cnt)
    return new


def draw_cum_reward(models, batch_size, dataset, n_class, n_dim, output_path, timestamp):
    rcParams['figure.figsize'] = 25, 15
    lwd = 5
    cmap = plt.get_cmap('tab20')
    colors=plt.cm.tab20(np.linspace(0, 1, 20))

    ax = plt.subplot(111)

    for e, model in enumerate(models):
        lst_rewards = model.lst_rewards
        plt.plot(get_cum_reward(lst_rewards), label=model.name, linewidth=lwd, color=colors[e])



    plt.plot(np.repeat(y.sum(axis=0).max(), len(lst_rewards)), label="Overall Best Arm (no context)", linewidth=lwd,color=colors[1],ls='dashed')
    #plt.plot(np.repeat(np.sign(y.sum(1)).sum(), len(lst_rewards[0])), label="Optimal Arm (no context)", linewidth=lwd,color=colors[1],ls='dashed')


    # import warnings
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 1.25])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
              fancybox=True, ncol=3, prop={'size':20})


    plt.tick_params(axis='both', which='major', labelsize=25)
    plt.xticks([i*9 for i in range(4)], [i*2 for i in range(4)])


    plt.xlabel(f'Rounds (models were updated every {batch_size} rounds)', size=30)
    plt.ylabel('Cumulative Mean Reward', size=30)
    plt.title(f'Comparison of Online Contextual Bandit Policies\n(Base Algorithm is Logistic Regression)\n\n{dataset} Dataset\n({n_class} categories, {n_dim} attributes)',size=30)
    plt.grid()
    plt.show()
    plt.savefig(f"{output_path}/{timestamp}_cum_reward.png")
    

def draw_train_loss(models, output_path, timestamp):
    
    min_length = 100000000
    for model in models:
        if "train_loss" in model.history:
            min_length = min(min_length, model.history["train_loss"])
        elif "d_train_loss" in model.history:
            min_length = min(min_length, model.history["d_train_loss"])
        else:
            raise Exception("No train loss.")
            
    plt.figure(figsize=(20, 20))
    for model in models:
        if "train_loss" in model.history:
            key = "train_loss"
        elif "d_train_loss" in model.history:
            key = "d_train_loss"

        length = len(model.history[key])
        div = length // min_length
        plt.plot(model.history[key][::div], label=model.name)
    plt.legned()
    plt.show()
    plt.savefig(f"{output_path}/{timestamp}_train_loss.png")
    
    
def draw_back_optimality(model, output_path, timestamp):
    acc = []
    real = []
    back = []
    cnt = 0
    for report in models[0].report_list[:-1]:
        tmp = []
        for p_vec, b_idx in zip(report.predict, report.back):
            p_idx = np.argmax(p_vec)
            acc.append(p_idx == b_idx)
            real.append(y[cnt][p_idx])
            back.append(y[cnt][b_idx])
            cnt += 1
            
    error = [abs(i-j) for i, j in zip(real, back)]

    plt.plot(np.cumsum(real), label="real")
    plt.plot(np.cumsum(back), label="back")
    plt.plot(np.cumsum(error), label="error")
    plt.legend()
    plt.show()
    plt.savefig(f"{output_path}/{timestamp}_{model.name}_back_optimality.png")
