####################################################

#      Some data visualization

####################################################

from tkinter import font
from turtle import onclick
import pandas as pd
import matplotlib.pyplot as plt
# from statsmodels.nonparametric.smoothers_lowess import lowess
from os import walk
from tsmoothie.smoother import *
import argparse
from matplotlib.colors import TABLEAU_COLORS


parser = argparse.ArgumentParser(description='Single agent reinforcement learning example script using TakeoffAviary')
parser.add_argument('--exp',                           type=str,            help='The experiment folder written as ./results/save-<env>-<algo>-<obs>-<act>-<time_date>', metavar='')
ARGS = parser.parse_args()
models_dir = ARGS.exp


if True: 
    filenames = next(walk(models_dir))[2]
    filenames.sort()
    # print(*list(filenames),sep='\n')
    # algo_names = [x.replace('run-', '').replace('-tag-eval_mean_reward.csv', '') for x in filenames]
    algo_names = [x.replace('run-', '').replace('-tag-train_value_loss.csv', '') for x in filenames]

    print("Please type the algorithm index to continue training")
    print(*list(enumerate(algo_names)),sep='\n')
    alg_index = input("(seperate indexes with comma or type 'all')\n")
    algo_index = [int(x) for x in alg_index.split(",")] if alg_index != "all" else [x for x in range(len(algo_names))]
    color_map = ["r","r","r","r","r","b","b","b","b","b"]#list(TABLEAU_COLORS.values())

    fig = plt.figure()
    for i in range(len(algo_index)):
        df = pd.read_csv(models_dir + filenames[i])
        df = df.to_numpy()
        
        smoothed_df = ConvolutionSmoother(window_len=50, window_type='ones')
        smoothed_df.smooth(df[:,2])
        # generate intervals
        low, up = smoothed_df.get_intervals('sigma_interval', n_sigma=.5)

        # plot the smoothed timeseries with intervals
        # plt.figure(figsize=(11,6))
        # plt.plot(df[:,1], smoothed_df.data[0], color='orange')
        # t = (df[:,0] - df[0,0])/3600
        t = (df[:,1])/1000000
        plt.plot(t, smoothed_df.smooth_data[0],color = color_map[i], linewidth = 2)
        plt.fill_between(t, low[0], up[0], alpha=0.05, color = color_map[i])
        # plt.plot(df[:,1], smoothed_df.smooth_data[0],color = color_map[i])
        # plt.fill_between(df[:,1], low[0], up[0], alpha=0.1)


    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color = color_map[0], linewidth = 2,  label='PPO'),
                       Line2D([0], [0], color = color_map[5], linewidth = 2,  label='A2C')]

    ax = plt.gca()
    ax.legend(handles=legend_elements,fontsize = 35)

    plt.grid(True)
    # plt.xlabel("Steps", fontsize = 15)
    plt.xlabel("Training steps (million step)", fontsize = 40)
    plt.ylabel("Validation reward", fontsize = 40)
    ax.tick_params(axis='both', which='major', labelsize=30)

    plt.title("Validation reward tracking over training process", fontsize = 40)

    # ax.set_rasterized(True)
    fig.set_size_inches(12*2, 7*2)
    plt.savefig('validation.png')
    # plt.savefig('value_loss.png')
    plt.show()
exit(0)

if True:
    l_f_a2c = np.load("/home/lin/from_remote/experiments/learning/1000_data/traj_length_f_A2C.npy", allow_pickle=True)
    l_s_a2c = np.load("/home/lin/from_remote/experiments/learning/1000_data/traj_length_s_A2C.npy", allow_pickle=True)
    l_f_ppo = np.load("/home/lin/from_remote/experiments/learning/1000_data/traj_length_f_PPO.npy", allow_pickle=True)
    l_s_ppo = np.load("/home/lin/from_remote/experiments/learning/1000_data/traj_length_s_PPO.npy", allow_pickle=True)


    fig = plt.figure(figsize =(10, 7))

    # Creating axes instance
    # ax = fig.add_axes([0, 0, 1, 1])
    plt.boxplot([l_s_a2c, l_s_ppo, l_f_a2c, l_f_ppo])
    plt.xticks([1, 2, 3, 4], ['Success A2C', 'Success PPO', 'Failed A2C', 'Failed PPO'], fontsize = 15)
    plt.ylabel("Step (0.1s/step)", fontsize = 15)
    plt.show()

    # exit()
    fig = plt.figure()
    training_duration = [
                        6*60+44, 6*60+44, 6*60+44,  # ppo_no_opt
                        3*60+57, 5*60+8, 4*60+25,   # a2c_no_opt
                        7*60+48, 7*60+18, 6*60+54]  # trpo
                        

    success_rate = [
                    49.84, 50.96, 45.84,            # ppo_no_opt
                    4.7, 68.04, 41.54,              # a2c_no_opt
                    1.14, 0.12, 0.76]               # trpo
                    
    test_list = ['PPO', 
                'A2C', 'TRPO']


    # import matplotlib.colors as mcolors
    from matplotlib.colors import TABLEAU_COLORS
    # color_cycle = ["r","r","r","g","g","g","b","b","b","c","c","c", "m", "m","m"]
    color_list = list(TABLEAU_COLORS.values())
    color_cycle = [x for x in color_list for i in range(3)][0:len(success_rate)]
    # print(color_cycle)
    plt.scatter(training_duration, success_rate, marker="o", color=color_cycle)


    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', lw=0,label=test_list[i], markerfacecolor=color_list[i], markeredgecolor=color_list[i]) for i in range(len(test_list))]

    ax = plt.gca()
    ax.legend(handles=legend_elements, fontsize = 12)
    plt.xlabel("Training duration (min)", fontsize = 15)
    plt.ylabel("Success rate(%)", fontsize = 15)
    plt.title("Resulting policy from different algorithms", fontsize = 15)
    plt.grid()
    fig.set_size_inches(6, 4.5)
    plt.savefig('trail_test.eps', format='eps', bbox_inches='tight')
    plt.show()

    training_duration = [4*60+59, 3*60+13, 4*60+6,  # id_ppo
                        10*60+4, 7*60+22, 10*60+39, # ppo
                        6*60+44, 6*60+44, 6*60+44,  # ppo_no_opt
                        3*60+57, 4*60+14, 3*60+11,  # a2c
                        3*60+57, 5*60+8, 4*60+25,   # a2c_no_opt
                        7*60+48, 7*60+18, 6*60+54]  # trpo
                        

    success_rate = [55.42, 0.16, 0.0,               # id_ppo
                    55.14, 44.48, 52.98,            # ppo
                    49.84, 50.96, 45.84,            # ppo_no_opt
                    40.48, 32.5, 0.84,              # a2c
                    4.7, 68.04, 41.54,              # a2c_no_opt
                    1.14, 0.12, 0.76]               # trpo
                    
    test_list = ['Aligned PPO', 'PPO', 'PPO no opt',
                'A2C', 'A2C no opt', 'TRPO']


    # import matplotlib.colors as mcolors
    from matplotlib.colors import TABLEAU_COLORS
    # color_cycle = ["r","r","r","g","g","g","b","b","b","c","c","c", "m", "m","m"]
    color_list = list(TABLEAU_COLORS.values())
    color_cycle = [x for x in color_list for i in range(3)][0:len(success_rate)]
    # print(color_cycle)
    plt.scatter(training_duration, success_rate, marker="o", color=color_cycle)


    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', lw=0,label=test_list[i], markerfacecolor=color_list[i], markeredgecolor=color_list[i]) for i in range(len(test_list))]

    ax = plt.gca()
    ax.legend(handles=legend_elements, fontsize = 12)
    plt.xlabel("Training duration (min)", fontsize = 12)
    plt.ylabel("Success rate over 5000 test (%)", fontsize = 12)
    plt.title("Training results from different algorithms, over 10 million steps", fontsize = 12)
    plt.grid()
    plt.show()

    # training_duration = [4*60+10, 4*60+9,  4*60+11, 4*60+21, 5*60+12,   #n_epoch=1, batch_size=25
    #                      3*60+36, 3*60+35, 3*60+35, 4*60+36, 3*60+57,   #n_epoch=1, batch_size=50  
    #                      3*60+24, 3*60+23, 3*60+25, 3*60+48 ]   #n_epoch=1, batch_size=75

    # success_rate = [70.08, 24.54, 75.34, 77.40, 69.78,               #n_epoch=1, batch_size=25
    #                 55.42, 55.62, 59.92, 65.00, 63.54,              #n_epoch=1, batch_size=50
    #                 49.10, 19.10, 36.78, 40.48]               #n_epoch=1, batch_size=75
    test_list = ['PPO batch size = 5','PPO batch size = 15','PPO batch size = 25', 'PPO batch size = 50', 'PPO batch size = 75']

    training_duration = [8*60+34, 8*60+29, 8*60+45,    #n_epoch=1, batch_size=5
                        5*60+3,  5*60+2,  5*60+2,     #n_epoch=1, batch_size=15
                        4*60+10, 4*60+9,  4*60+11,    #n_epoch=1, batch_size=25
                        3*60+36, 3*60+35, 3*60+35,    #n_epoch=1, batch_size=50  
                        3*60+24, 3*60+23, 3*60+25 ]   #n_epoch=1, batch_size=75

    success_rate = [ 0.44, 61.42, 62.02,               #n_epoch=1, batch_size=5
                    22.38, 74.48, 34.70,               #n_epoch=1, batch_size=15
                    70.08, 24.54, 75.34,               #n_epoch=1, batch_size=25
                    55.42, 55.62, 59.92,               #n_epoch=1, batch_size=50
                    49.10, 19.10, 36.78]               #n_epoch=1, batch_size=75

    # import matplotlib.colors as mcolors
    from matplotlib.colors import TABLEAU_COLORS
    # color_cycle = ["r","r","r","g","g","g","b","b","b","c","c","c", "m", "m","m"]
    color_list = list(TABLEAU_COLORS.values())
    color_cycle = [x for x in color_list for i in range(3)][0:len(success_rate)]
    # print(color_cycle)
    plt.scatter(training_duration, success_rate, marker="o", color=color_cycle)

    # training_duration = [4*60+21, 5*60+12,   #n_epoch=1, batch_size=25
    #                      4*60+36, 3*60+57,   #n_epoch=1, batch_size=50  
    #                      3*60+48 ]   #n_epoch=1, batch_size=75

    # success_rate = [77.40, 69.78,               #n_epoch=1, batch_size=25
    #                 65.00, 63.54,              #n_epoch=1, batch_size=50
    #                 40.48]               #n_epoch=1, batch_size=75
    # color_cycle = [x for x in color_list[2:-1] for i in range(2)][0:len(success_rate)]
    # plt.scatter(training_duration, success_rate, marker="x", color=color_cycle)


    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', lw=0,label=test_list[i], markerfacecolor=color_list[i], markeredgecolor=color_list[i]) for i in range(len(test_list))]

    ax = plt.gca()
    ax.legend(handles=legend_elements)
    plt.xlabel("Training duration (min)")
    plt.ylabel("Success rate over 5000 test (%)")
    plt.title("Training results from different algorithms, over 10 million steps")
    plt.grid()
    plt.show()


    test_list = ['PPO on GPU','PPO on CPU','A2C on GPU','A2C on CPU']
    training_duration = [4*60+10, 4*60+9,   4*60+11,   #ppo, gpu
                        3*60+3 , 2*60+32,  2*60+15,   #ppo, cpu
                        3*60+57, 4*60+14, 3*60+11,    #a2c, gpu
                        2*60+43, 2*60+45, 2*60+44]    #a2c, cpu

    success_rate = [70.08, 24.54, 75.34,             #ppo, gpu
                    73.84, 71.46, 69.82,             #ppo, cpu
                    40.48, 32.5,  0.84,               #a2c, gpu
                    32.98, 65.94, 53.46]               #a2c, cpu
    marker_cycle = ["o", "x", "o", "x"]
    color_cycle = ["r", "r", "b", "b"]
    for i in range(len(marker_cycle)):
        plt.scatter(training_duration[3*i:3*i+3], success_rate[3*i:3*i+3], marker=marker_cycle[i], color=color_cycle[i])


    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker=marker_cycle[i], lw=0,label=test_list[i], markerfacecolor=color_cycle[i], markeredgecolor=color_cycle[i]) for i in range(len(test_list))]

    ax = plt.gca()
    ax.legend(handles=legend_elements)
    plt.xlabel("Training duration (min)")
    plt.ylabel("Success rate over 5000 test (%)")
    plt.title("Training results from different algorithms, over 10 million steps")
    plt.grid()
    plt.show()

    test_list = ['PPO with Adam','PPO with RMSprop','A2C with RMSprop']
    training_duration = [3*60+3 , 2*60+32,  2*60+15,   #ppo, cpu, adam
                        2*60+59, 2*60+39,  2*60+11,   #ppo, cpu, rms
                        2*60+43, 2*60+45,  2*60+44]   #a2c, cpu, rms

    success_rate = [73.84, 71.46, 69.82,    #ppo, cpu, adam
                    72.48, 75.82, 67.66,    #ppo, cpu, rms
                    32.98, 65.94, 53.46]    #a2c, cpu, rms
    color_cycle = ["r", "b", "c"]
    for i in range(len(color_cycle)):
        plt.scatter(training_duration[3*i:3*i+3], success_rate[3*i:3*i+3], marker="o", color=color_cycle[i])


    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker="o", lw=0,label=test_list[i], markerfacecolor=color_cycle[i], markeredgecolor=color_cycle[i]) for i in range(len(test_list))]

    ax = plt.gca()
    ax.legend(handles=legend_elements)
    plt.xlabel("Training duration (min)")
    plt.ylabel("Success rate over 5000 test (%)")
    plt.title("Training results from different algorithms, over 10 million steps")
    plt.grid()
    plt.show()


training_duration = [60.26, 62.68, 47.94,
                     51.74, 40.48, 40.45,
                     55.83, 51.35, 46.64,
                     63.60, 67.32, 59.38,
                     58.02, 50.83, 66.57,
                     47.88, 57.08, 45.82,
                     58.39, 51.26, 41.84,
                     52.83, 48.26, 57.85]

                    

success_rate = [88.70, 91.20, 70.48,
                83.74, 60.86, 60.74,
                76.50, 78.30, 81.56,
                86.02, 87.14, 82.62,
                51.20, 56.74, 28.60,
                61.98, 54.98, 59.84,
                55.34, 61.72, 53.30,
                50.98, 57.10, 57.66]

                
test_list = ['A2C', 'PPO']

fig = plt.figure()
# import matplotlib.colors as mcolors
from matplotlib.colors import TABLEAU_COLORS
# color_cycle = ["b","b","b","b","r","r", "r", "r"]
# color_list = list(TABLEAU_COLORS.values())
color_list = ["b", "r"]
color_cycle = [x for x in color_list for i in range(12)][0:len(success_rate)]
# print(color_cycle)
plt.scatter(training_duration, success_rate, marker="o", color=color_cycle)


from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], marker='o', lw=0,label=test_list[i], markerfacecolor=color_list[i], markeredgecolor=color_list[i]) for i in range(len(test_list))]


ax = plt.gca()
ax.legend(handles=legend_elements, fontsize = 15)
plt.xlabel("Average trajectory duration (step)", fontsize = 18)
plt.ylabel("Success rate over (%)", fontsize = 18)
plt.title("Policy time optimality vs success rate", fontsize = 20)
plt.grid()
# ax.axis('square')
plt.xlim([40, 70])
plt.ylim([50, 95])
ax.tick_params(axis='both', which='major', labelsize=15)
fig.set_size_inches(10, 5)
plt.savefig('time_success.eps', format='eps', bbox_inches='tight')
plt.show()
