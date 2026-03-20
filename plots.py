import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

OUTPUT_DIR = './images/'

def format_num(pct, allvals):
    absolute = int(np.round(pct/100*np.sum(allvals)))
    return f"{absolute:d}\n({pct:.1f}%)"

def plot_triple_class_dist(df_dict, suptitle, filename):
    fig, axes = plt.subplots(1, 3, figsize=(4 * 3, 5.2), subplot_kw={'aspect': 'equal'})
    fig.suptitle(suptitle, fontweight='bold')

    for i, (model, df) in enumerate(df_dict.items()):
        ax = axes[i]

        data = df.values
        label = df.index

        wedges, texts, autotexts = ax.pie(
            x=data, 
            autopct=lambda pct: format_num(pct, data), 
            textprops={'color': 'black'}, 
            pctdistance=0.37*(i+1),
        )
        ax.set_title(f'{model}')
        ax.legend(wedges, label, fontsize=10,
            loc='upper center', 
            ncol=(2 if model.lower()=='main model' else 1),
            bbox_to_anchor=(0.5, 0.08, 0, 0)
        )
        plt.setp(autotexts, size=10, weight='normal')
    
    plt.tight_layout() 
    plt.savefig(os.path.join(OUTPUT_DIR, f'{filename}.png'))
    plt.show()

def plot_data_split(df, title, filename):
    data = df.values
    labels = df.index
    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw={'aspect': 'equal'})
    wedges, texts, autotexts = ax.pie(
        x=data, 
        autopct=lambda pct: format_num(pct, data), 
        textprops={'color': 'black'}, 
        pctdistance=0.5,
    )
    ax.set_title(title, fontweight='bold', fontsize=12)
    ax.legend(wedges, labels, 
        loc='lower center', 
        bbox_to_anchor=(0.5, -0.2, 0, 0)
    )
    plt.setp(autotexts, size=10, weight='normal')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'{filename}.png'))
    plt.show()

def plot_triple_data_split(df_list, suptitle, filename):
    n = len(df_list)
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 4.2), subplot_kw={'aspect': 'equal'})
    fig.suptitle(suptitle, fontweight='bold')

    for i, (df, title) in enumerate(df_list):
        ax = axes[i]
        if df.empty:
            ax.set_title(title + "\n(No Data)")
            ax.axis('off')
            continue

        data = df.values
        label = df.index
        if np.sum(data) == 0:
            ax.set_title(title + "\n(No Data)")
            ax.axis('off')
            continue

        wedges, texts, autotexts = ax.pie(
            x=data, 
            autopct=lambda pct: format_num(pct, data), 
            textprops={'color': 'black'}, 
            pctdistance=0.5,
        )
        ax.set_title(title)
        ax.legend(wedges, label, 
            loc='lower center', 
            bbox_to_anchor=(0.5, -0.2, 0, 0)
        )
        plt.setp(autotexts, size=10, weight='normal')

    plt.tight_layout() 
    plt.savefig(os.path.join(OUTPUT_DIR, f'{filename}.png'))
    plt.show()

def plot_confusion_matrix(cm, class_names, figsize, cmap, filename):
    plt.figure(figsize=figsize)
    p = sns.heatmap(cm, annot=True, fmt='d', annot_kws={'fontweight':'bold'}, cmap=cmap, 
                    xticklabels=class_names, yticklabels=class_names)
    p.set_xticklabels(p.get_xticklabels(), rotation=90, fontsize = 10)
    p.set_yticklabels(p.get_yticklabels(), rotation=0, fontsize = 10)
    plt.xlabel('Predicted Class', fontweight='bold')
    plt.ylabel('Actual Class', fontweight='bold')
    plt.title('Confusion Matrix', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'{filename}.png'))
    plt.show()

def plot_learning_rate(arr, title, filename):
    max_value = max(arr)
    plt.figure(figsize=(7, 4.375))
    plt.plot(range(1, len(arr)+1), arr, '.-', label='Learning Rate')
    plt.legend()
    plt.title(title, fontsize=12, fontweight='bold')
    plt.xlabel('Epoch', fontweight='bold')
    plt.ylabel('Value', fontweight='bold')
    plt.xlim([-0.02*len(arr), 1.02*len(arr)])
    plt.ylim([-0.02*max_value, 1.02*max_value])
    plt.tight_layout()
    plt.grid()
    plt.savefig(os.path.join(OUTPUT_DIR, f'{filename}.png'))
    plt.show()

def plot_triple_model(arr_binary, arr_quinary, arr_main, title, filename):
    arr = np.concatenate((arr_binary, arr_quinary, arr_main))
    n = max(len(arr_binary), len(arr_quinary), len(arr_main))
    max_value = max(arr)
    plt.figure(figsize=(7, 4.375))
    plt.plot(range(1, len(arr_binary)+1), arr_binary, '.-', label='Binary Model')
    plt.plot(range(1, len(arr_quinary)+1), arr_quinary, '.-', label='Quinary Model')
    plt.plot(range(1, len(arr_main)+1), arr_main, '.-', label='Main Model')
    plt.legend()
    plt.title(title, fontsize=12, fontweight='bold')
    plt.xlabel('Epoch', fontweight='bold')
    plt.ylabel('Value', fontweight='bold')
    plt.xlim([-0.02*n, 1.02*n])
    plt.ylim([-0.02*max_value, 1.02*max_value])
    plt.tight_layout()
    plt.grid()
    plt.savefig(os.path.join(OUTPUT_DIR, f'{filename}.png'))
    plt.show()

def dual_plot_triple_model(suptitle, filename, 
                           train_arr_binary, train_arr_quinary, train_arr_main, title1, 
                           val_arr_binary, val_arr_quinary, val_arr_main, title2):
    
    train_arr = np.concatenate((train_arr_binary, train_arr_quinary, train_arr_main))
    val_arr = np.concatenate((val_arr_binary, val_arr_quinary, val_arr_main))
    train_n = max(len(train_arr_binary), len(train_arr_quinary), len(train_arr_main))
    val_n = max(len(val_arr_binary), len(val_arr_quinary), len(val_arr_main))
    train_max_value = max(train_arr)
    val_max_value = max(val_arr)
    
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax = ax.flatten()
    
    fig.suptitle(suptitle, fontweight='bold', fontsize=14)
    
    ax[0].plot(range(1, len(train_arr_binary)+1), train_arr_binary, label='Binary Model')
    ax[0].plot(range(1, len(train_arr_quinary)+1), train_arr_quinary, label='Quinary Model')
    ax[0].plot(range(1, len(train_arr_main)+1), train_arr_main, label='Main Model')
    ax[0].legend()
    ax[0].set_title(title1, fontsize=12, fontweight='bold')
    ax[0].set_xlabel('Epoch', fontweight='bold')
    ax[0].set_ylabel('Value', fontweight='bold')
    ax[0].set_xlim([-0.02*train_n, 1.02*train_n])
    ax[0].set_ylim([-0.02*train_max_value, 1.02*train_max_value])
    ax[0].grid()
    
    ax[1].plot(range(1, len(val_arr_binary)+1), val_arr_binary, label='Binary Model')
    ax[1].plot(range(1, len(val_arr_quinary)+1), val_arr_quinary, label='Quinary Model')
    ax[1].plot(range(1, len(val_arr_main)+1), val_arr_main, label='Main Model')
    ax[1].legend()
    ax[1].set_title(title2, fontsize=12, fontweight='bold')
    ax[1].set_xlabel('Epoch', fontweight='bold')
    ax[1].set_ylabel('Value', fontweight='bold')
    ax[1].set_xlim([-0.02*val_n, 1.02*val_n])
    ax[1].set_ylim([-0.02*val_max_value, 1.02*val_max_value])
    ax[1].grid()

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'{filename}.png'))
    plt.show()

def plot_training_log(train_arr, val_arr, title, filename):
    max_value = max(max(train_arr), max(val_arr))
    plt.figure(figsize=(7, 4.375))
    plt.plot(range(1, len(train_arr)+1), train_arr, '.-', label='Train')
    plt.plot(range(1, len(val_arr)+1), val_arr, '.-', label='Validation')
    plt.legend()
    plt.title(title, fontsize=12, fontweight='bold')
    plt.xlabel('Epoch', fontweight='bold')
    plt.ylabel('Value', fontweight='bold')
    plt.xlim([-0.02*len(train_arr), 1.02*len(train_arr)])
    plt.ylim([-0.02*max_value, 1.02*max_value])
    plt.tight_layout()
    plt.grid()
    plt.savefig(os.path.join(OUTPUT_DIR, f'{filename}.png'))
    plt.show()

def dual_plot_training_log(suptitle, filename, train_arr1, val_arr1, title1, train_arr2, val_arr2, title2):
    max_value1 = max(max(train_arr1), max(val_arr1))
    max_value2 = max(max(train_arr2), max(val_arr2))
    
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax = ax.flatten()
    
    fig.suptitle(suptitle, fontweight='bold', fontsize=14)
    
    ax[0].plot(train_arr1, label='Train')
    ax[0].plot(val_arr1, label='Validation')
    ax[0].legend()
    ax[0].set_title(title1)
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Value')
    ax[0].set_xlim([-0.02*len(train_arr1), 1.02*len(train_arr1)])
    ax[0].set_ylim([-0.02*max_value1, 1.02*max_value1])
    ax[0].grid()
    
    ax[1].plot(train_arr2, label='Train')
    ax[1].plot(val_arr2, label='Validation')
    ax[1].legend()
    ax[1].set_title(title2)
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Value')
    ax[1].set_xlim([-0.02*len(train_arr2), 1.02*len(train_arr2)])
    ax[1].set_ylim([-0.02*max_value2, max(1.02, 1.02*max_value2)])
    ax[1].grid()

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'{filename}.png'))
    plt.show()
