import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def analyse(csvfile):
    data = pd.read_csv(csvfile).dropna(inplace=False)
    title = csvfile.split('_')[0]
    print(f'Analysing {title}')
    intensities = ["LOW","MEDIUM","HIGH"]
    counts = [[0]*3 for i in range(3)]
    for i in range(3):
        for j in range(3):
            counts[i][j] = sum((data['issue_severity'] == intensities[i]) & (data['issue_confidence'] == intensities[j]))
    plt.imshow(counts,cmap='Reds')
    plt.xlabel('Confidence')
    plt.ylabel('Severity')
    plt.xticks(np.arange(3), intensities)
    plt.yticks(np.arange(3), intensities)
    for i in range(3):
        for j in range(3):
            plt.text(j, i, counts[i][j], ha='center', va='center', color='black')
    plt.title(f'{title} : {len(data)} issues from {len(data["commit SHA"].unique())} commits')
    plt.savefig(f'{csvfile}_heatmap.png')
    plt.close()
    plt.bar(intensities, [sum(data['issue_severity'] == i) for i in intensities])
    plt.title(f'{title} severity distribution')
    plt.xlabel('Severity')
    plt.ylabel('Frequency')
    plt.savefig(f'{csvfile}_severity.png')
    plt.close()
    plt.bar(intensities, [sum(data['issue_confidence'] == i) for i in intensities])
    plt.title(f'{title} confidence distribution')
    plt.xlabel('Confidence')
    plt.ylabel('Frequency')
    plt.savefig(f'{csvfile}_confidence.png')
    plt.close()
    plt.hist(data['test_id'],bins=np.arange(1,1+len(data['test_id'].unique()),1))
    xt = np.array(plt.gca().get_xticks()) + 0.5
    xtl = plt.gca().get_xticklabels()
    plt.xticks(xt,labels=xtl,rotation=90)
    plt.xlabel('Bandit Test ID')
    plt.ylabel('Frequency')
    plt.title(f'{title} CWE distribution')
    plt.savefig(f'{csvfile}_cwe.png')
    plt.close()
    print(f'Analysis of {title} complete')
    return data[["test_id","issue_cwe"]].drop_duplicates(inplace=False)

def badness_graph(csvfile):
    data = list(pd.read_csv(csvfile).values)
    title = csvfile.split('_')[0]
    #data = [x for x in data if (x[7] == "HIGH" or x[7] == "MEDIUM" or x[7] is None)]
    print(len(data))
    fileissues = {}
    S = np.array([0]*3)
    T = 0
    Slist = []
    Tlist = []
    TClist = [0]
    Clist = []
    last_commit = None
    cum = data[0][2][:4]
    for i in range(len(data)):
        commit = data[i][2]
        cum = commit[:4]
        h = (data[i][7] == "HIGH")
        m = (data[i][7] == "MEDIUM")
        l = (data[i][7] == "LOW")
        toadd = np.array([h,m,l])
        if data[i][6] is None:
            Slist.append(S)
            Tlist.append(T)
            if fileissues[data[i][0]][0] > (0.2*S)[0] or fileissues[data[i][0]][1] > (0.2*S)[1] or fileissues[data[i][0]][2] > (0.2*S)[2]: 
                TClist.append(T)
                Clist.append(cum)
            S = S - fileissues[data[i][0]]
            fileissues[data[i][0]] = 0
        elif data[i][0] not in fileissues:
            Slist.append(S)
            Tlist.append(T)
            fileissues[data[i][0]] = toadd
            S = S + toadd
        elif last_commit != commit:
            Slist.append(S)
            Tlist.append(T)
            if fileissues[data[i][0]][0] > (0.2*S)[0] or fileissues[data[i][0]][1] > (0.2*S)[1] or fileissues[data[i][0]][2] > (0.2*S)[2]: 
                TClist.append(T)
                Clist.append(cum)
            S = S - fileissues[data[i][0]] + toadd
            fileissues[data[i][0]] = toadd
        else:
            fileissues[data[i][0]] = fileissues[data[i][0]] + toadd
            S = S + toadd
        if commit != last_commit:T += 1
        last_commit = commit
    Slist = np.array(Slist)
    plt.plot(Tlist,Slist[:,0],label='High')
    plt.plot(Tlist,Slist[:,1],label='Medium')
    plt.plot(Tlist,Slist[:,2],label='Low')
    plt.legend()
    plt.xlabel('Commit')
    plt.ylabel('High severity issues')
    plt.xticks(TClist,["start"]+Clist,rotation=90)
    plt.title(f'{title} Issues Graph')
    plt.savefig(f'{csvfile}_badness.png')
    plt.close()
    print(f'Badness graph of {title} complete')

if __name__ == "__main__":
    files = [
        "NumPy_results/commits_info.csv",
        "OpenCV_results/commits_info.csv",
        "PyTorch_results/commits_info.csv"
        ]
    S = []
    for csvfile in files:
        title = csvfile.split('_')[0]
        S.append("#### " + title + " :")
        cwes = sorted([f"[{x[0]}]({x[1]})" for x in analyse(csvfile).values])
        S.append(", ".join(cwes))
    f = open("analysis_output.md","w")
    f.write("\n".join(S))
    f.close()
    print("Analysis complete")
    for csvfile in files:badness_graph(csvfile)
    print("Badness graphs complete")