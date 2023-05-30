import numpy as np
import matplotlib.pyplot as plt
from pylab import text
  
def plot_evaluation(many,lines,filename):
  #plot function for evaluation of performance
  fig, ax1 = plt.subplots(figsize=(8,5))
  note='Mean : ' + str(np.mean(lines))+'\nMax : '+ str(np.max(lines))+'\nMin : '+str(np.min(lines))
  ax1.plot(lines)
  ax1.axhline(y=np.mean(lines),color='k', linestyle='--',label='Average Lines Cleared')
  ax1.legend(loc=5, borderaxespad=0.)
  ax1.text(3, 500, note, style='italic', fontsize=12,
        bbox={'facecolor': 'grey', 'alpha': 0.5, 'pad': 10})
  ax1.set(xlabel='Episode Number', ylabel='Lines Cleared',title='Lines Cleared in '+str(many)+' games, with cap 1000 lines')
  plt.grid()
  plt.savefig(filename)

def refined_plot(steps, scores, epsilons, filename,evry=100,y_lbl='Score'):
    x= [i for i in range(0,len(scores),evry)]
    eps=[epsilons[i] for i in x]
    frame=[steps[i] for i in x]
    fig=plt.figure(figsize=(8,5))
    #fig, (ax, ax3) = plt.subplots(1, 2)
    #fig, axes = plt.subplots(nrows=2, ncols=1)
    ax=fig.add_subplot(111, label="1")
    ax2=fig.add_subplot(111, label="2", frame_on=False)

    ax.plot(frame, eps, color="C0")
    ax.set_xlabel("Training Steps", color="k")
    ax.set_ylabel("Epsilon", color="C0")
    ax.tick_params(axis='x', colors="k")
    ax.tick_params(axis='y', colors="C0")

    N = len(x)
    running_avg = np.empty(N)
    for i,t in enumerate(x):
	    running_avg[i] = np.mean(scores[max(0, t-evry):(t+1)])

    ax2.plot(frame, running_avg, color="k")
    ax2.axes.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    ax2.set_ylabel(y_lbl, color="k")
    ax2.yaxis.set_label_position('right')
    ax2.tick_params(axis='y', colors="k")
    ax2.set_title('Average Lines Cleared per 1000 episodes')
    plt.grid()
    plt.savefig(filename)
    
def modified_plot(steps, scores, epsilons, filename,evry=100,y_lbl='Score' ,lines=None):
    x= [i for i in range(0,len(scores),evry)]
    eps=[epsilons[i] for i in x]
    frame=[steps[i] for i in x]
    fig=plt.figure()
    ax=fig.add_subplot(111, label="1")
    ax2=fig.add_subplot(111, label="2", frame_on=False)

    ax.plot(frame, eps, color="C0")
    ax.set_xlabel("Training Steps", color="C0")
    ax.set_ylabel("Epsilon", color="C0")
    ax.tick_params(axis='x', colors="C0")
    ax.tick_params(axis='y', colors="C0")

    N = len(x)
    running_avg = np.empty(N)
    for i,t in enumerate(x):
	    running_avg[i] = np.mean(scores[max(0, t-evry):(t+1)])

    ax2.scatter(frame, running_avg, color="C1")
    ax2.axes.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    ax2.set_ylabel(y_lbl, color="C1")
    ax2.yaxis.set_label_position('right')
    ax2.tick_params(axis='y', colors="C1")

    plt.savefig(filename)

def simple_plot(steps,var,filename,evry=100,name='Score'):
  x= [i for i in range(0,len(var),evry)]

  frame=[steps[i] for i in x]
  fig=plt.figure()
  N = len(x)
  running_avg = np.empty(N)
  for i,t in enumerate(x):
    running_avg[i] = np.mean(var[max(0, t-evry):(t+1)])

  plt.scatter(frame, running_avg, color="C1")
  plt.ylabel(name, color="C1")
  plt.xlabel('Training Steps')
  plt.savefig(filename)


def plot_hyperparam(lines,gamma,lr,avg,Root,many=1000):
  filename1=Root+'training.png'
  filename2=Root+'evaluation.png'
  colors=['tab:blue','tab:orange','tab:green','tab:red','tab:purple']
  fig, ax1 = plt.subplots(figsize=(12,6))
  for i in range(len(lines)):
    x= [i for i in range(0,len(lines[i]),100)]
    avg_l = np.empty(len(x))
    for j,t in enumerate(x):
      avg_l[j]= np.mean(lines[i][max(0, t-100):(t+1)])
    lbl='Gamma : '+str(gamma[i])+', LR : '+str(lr[i])
    ax1.plot(x,avg_l,label=lbl,color=colors[i])
    #ax1.axhline(y=avg[i], linestyle='--',label='Average Lines Cleared in Evaluation')
  
  ax1.legend(loc=4, borderaxespad=0.)
  ax1.set(xlabel='Episode Number', ylabel='Lines Cleared',title='Average Lines Cleared each 1000 episodes during training')
  plt.grid()
  #plt.show()
  plt.savefig(filename1)

  fig, ax2 = plt.subplots(figsize=(10,6))
  lbl=['Gamma : '+str(gamma[i])+',\n LR : '+str(lr[i]) for i in range(len(lines))]
  ax2.bar(lbl,avg,color=colors)
  ax2.set(xlabel='Parameters Used', ylabel=' Average Lines Cleared',title='Average Lines Cleared in '+str(many)+' games in evaluation, with cap 1000 lines.')
  #plt.show()
  plt.savefig(filename2)
  

def boxplotting(avg_lines_cleared_ev,names,saveRoot):
    # Create an axes instance
    fig, ax= plt.subplots(figsize=(10,6))

    # Create the boxplot
    bp = ax.boxplot(avg_lines_cleared_ev,labels=names,showmeans=True)
    for line in bp['means']:
      # get position data for median line
      x, y = line.get_xydata()[0]  # top of median line
      # overlay median value
      text(x-0.1, y-15, '%.0f' % y, horizontalalignment='center')  # draw above, centered

    for box in bp['boxes']:
      x, y = box.get_path().vertices[1]  # bottom of left line
      text(x+0.03, y-15, '%.0f' % y, horizontalalignment='center',  # centered
      verticalalignment='top')      # below
      x, y = box.get_path().vertices[2]  # bottom of right line
      text(x+0.05, y+40, '%.0f' % y,
        horizontalalignment='center',  # centered
        verticalalignment='top')      # below
    ax.set(xlabel='Agent', ylabel='Lines Cleared',title='Lines Cleared in Evaluation of 1000 games')
    #plt.show()
    filename4=saveRoot+'BoxplotAll.png'
    plt.savefig(filename4)
