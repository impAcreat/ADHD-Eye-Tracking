import numpy as np
import json
import numpy as np
import matplotlib.pyplot as plt

def preprocess(data):
    print(f"data length: {len(data)}")
    label_freqs = {
        0: [[], [], []],
        1: [[], [], []]
    }
    for item in data:
        label = item["label"]
        for i in range(3):
            label_freqs[label][i].append(item["frequencies"][i])

    return label_freqs

def analysis():
    with open('files/result.json', 'r') as f:
        data = json.load(f)
    label_freqs = preprocess(data)

    print(f"pos: neg = {len(label_freqs[0][0])} : {len(label_freqs[1][0])}")

    for metric_idx in range(3):
        print(f"------ metric {metric_idx} ------")
        data_label0 = np.array(label_freqs[0][metric_idx])
        data_label1 = np.array(label_freqs[1][metric_idx])
        # 绘制双直方图
        plt.figure(figsize=(10, 6))
        plt.hist(data_label0, bins=30, alpha=0.5, color='blue', label='Label 0')
        plt.hist(data_label1, bins=30, alpha=0.5, color='red', label='Label 1')

        if metric_idx == 0:
            plt.title('Fixation', fontsize=16)
        elif metric_idx == 1:
            plt.title('Saccade', fontsize=16)
        elif metric_idx == 2:
            plt.title('PSO', fontsize=16)
        plt.xlabel('event proportion', fontsize=12)
        plt.ylabel('frequency', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(linestyle='--', alpha=0.4)
        plt.show()

# analysis()

def visualize_1list(data):
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=30, alpha=0.5, color='blue')
    plt.xlabel('value', fontsize=12)
    plt.ylabel('frequency', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(linestyle='--', alpha=0.4)
    plt.show()

def merge_continuous_events(events):
    merged = []
    current_event = events[0]
    start_idx = 0
    
    for i in range(1, len(events)):
        if events[i] != current_event:
            merged.append((current_event, start_idx, i-1))  # (事件类型, 开始索引, 结束索引)
            current_event = events[i]
            start_idx = i
    merged.append((current_event, start_idx, len(events)-1))  # 添加最后一个事件
    
    return merged

def analysis2():
    with open('files/result.json', 'r') as f:
        data = json.load(f)

    pos_avg = []
    neg_avg = []
    pos_count = []
    neg_count = []

    for each in data:
        events = each['result']
        merged_events = merge_continuous_events(events)

        counts = [0, 0, 0]
        durations = [0, 0, 0]
        specific_durations = [[], [], []]

        for event, start_idx, end_idx in merged_events:
            cur_duration = end_idx - start_idx + 1
            if cur_duration < 2:
                continue
            durations[event] += cur_duration
            counts[event] += 1
            specific_durations[event].append(cur_duration)

        for i in range(3):
            durations[i] = (durations[i] / counts[i]) if counts[i] != 0 else 0
        
        if each['label'] == 1:
            pos_avg.append(durations)
            pos_count.append(counts)
        else:
            neg_avg.append(durations)
            neg_count.append(counts)

        # print(f"-- {each['id']} is done")

        ## visualize
        # for i in range(3):
        #     visualize_1list(specific_durations[i])
        
        # break

    pos_avg = np.mean(pos_avg, axis=0)
    neg_avg = np.mean(neg_avg, axis=0)
    pos_count = np.mean(pos_count, axis=0)
    neg_count = np.mean(neg_count, axis=0)

    print(f"pos avg event duration: {pos_avg}")
    print(f"neg avg event duration: {neg_avg}")
    print(f"pos avg event count: {pos_count}")
    print(f"neg avg event count: {neg_count}")

analysis2()