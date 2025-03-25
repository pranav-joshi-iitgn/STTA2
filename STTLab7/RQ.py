import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit
files = ["NumPy_results/commits_info.csv","OpenCV_results/commits_info.csv","PyTorch_results/commits_info.csv"]

def to7tuple(x):
    return (
        x[1], # file path
        x[6], # test_id
        x[7], # issue_severity
        x[8], # issue_confidence
        int(x[10]), # col_offset
        int(x[11]), # end_col_offset
        len(x[12].split(",")) # length of line_range
        )

def get_pairs(file):
    data = list(pd.read_csv(file).values)
    data = [[x if x is not np.nan else None for x in y] for y in data]
    print(len(data))
    last_commit = None
    t = 0
    Done = []
    Current = {}
    IssuesFound = {}
    for x in data:
        commit = x[2]
        if commit != last_commit:
            # Complete issues that were not found in the last commit
            for file_path in Current:Current[file_path] = set(Current[file_path])
            torem = []
            for issue in IssuesFound:
                file_path = issue[0]
                if file_path not in Current:continue # The file was not modified in the last commit
                found = Current[file_path] # The issues found in the last commit
                if issue not in found:
                    Done.append(issue + IssuesFound[issue] + (last_commit,t))
                    torem.append(issue)
            for issue in torem:del IssuesFound[issue]
            # Add all issue gathered from the last commit to the IssuesFound
            S = 0
            for file_path in Current:
                S += len(Current[file_path])
                for issue in Current[file_path]:
                    IssuesFound[issue] = (last_commit,t)
            t += 1
            last_commit = commit
            Current = {}
        file_path = x[1]
        severity = x[7]
        test_id = x[6]
        if test_id is None:
            torem =[]
            for issue in IssuesFound:
                if issue[0] == file_path:
                    Done.append(issue + IssuesFound[issue] + (commit,t))
                    torem.append(issue)
            Current[file_path] = []
            for issue in torem:del IssuesFound[issue]
        else:
            y = to7tuple(x)
            if file_path in Current:Current[file_path].append(y)
            else: Current[file_path] = [y]
    return Done


def plot_pairs(csvfile,repo):
    data = pd.read_csv(csvfile)
    data = data.dropna()
    data = data.values
    severity = data[:,2]
    confidence = data[:,3]
    col_offset = data[:,4]
    end_col_offset = data[:,5]
    line_range = data[:,6]
    commit = data[:,7]
    commit_number = data[:,8]
    fix_commit = data[:,9]
    fix_commit_number = data[:,10]
    wait = fix_commit_number - commit_number
    # draw histograms for severity and confidence
    plt.figure()
    plt.subplot(1,2,1)
    plt.hist(severity)
    plt.title("Severity")
    plt.subplot(1,2,2)
    plt.hist(confidence)
    plt.title("Confidence")
    plt.savefig(csvfile[:-4]+"_fix_hist.png")
    # draw timeline for introduction and fixing of issues
    Map = {"LOW":1,"MEDIUM":2,"HIGH":3}
    plt.figure()
    plt.bar(commit_number,[Map[x] for x in severity],color = 'blue')
    plt.bar(fix_commit_number,[- Map[x] for x in severity],color = "red")
    plt.ylabel("Severity")
    plt.xlabel("Commit Number")
    plt.yticks([-3,-2,-1,1,2,3],["H","M","L","L","M","H"])
    plt.title(f"{repo} Issue Timeline")
    plt.grid(True)
    # show x axis
    plt.plot([0,fix_commit_number[-1]],[0,0],color = "black")
    plt.legend(["Timeline","Introduction","Fixing"])
    plt.ylim(-4,4)
    plt.savefig(csvfile[:-4]+"_timeline.png")
    # draw histogram for waiting time where severity is high,medium and low
    big = 0
    for s in ["LOW","MEDIUM"]:#High not included coz it has no issues
        plt.figure()
        w = wait[severity == s]
        plt.hist(w,label="frequency")
        plt.title(f"{repo} {s} severity waiting Time")
        plt.xlabel("Number of Commits")
        plt.ylabel("Frequency")
        # plot exponential fit
        P = plt.gca().patches
        x = np.array([p.get_x() for p in P])
        y = np.array([p.get_height() for p in P])
        def func(x, a, b):return a * np.exp(-b * x)
        popt, pcov = curve_fit(func, x, y)
        plt.plot(x, func(x, *popt), 'r-', label='y = %5.3f * exp(- %5.3f x )' % tuple(popt))
        plt.legend()
        plt.savefig(csvfile[:-4]+f"_wait_hist_{s}.png")
    plt.close()

if __name__ == "__main__":
    for file in files:
        print(file)
        pairs = get_pairs(file)
        print(len(pairs))
        with open(file[:-4]+"_pairs.csv","w") as f:
            f.write("file_path,test_id,issue_severity,issue_confidence,col_offset,end_col_offset,line_range,commit,commit_number,fix_commit,fix_commit_number\n")
            for pair in pairs:f.write(",".join([str(x) for x in pair])+"\n")

    print("Plotting")
    for file in files:
        plot_pairs(file[:-4]+"_pairs.csv",file.split("/")[0].split("_")[0])

    print("Done")