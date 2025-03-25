import sys,csv,os,pandas
from pydriller import Repository
def get_commits(url,name):
    columns = ["old_file path","new_file path","commit SHA","parent commit SHA","new_file_MCC"]
    band = ["test_name","test_id","issue_severity","issue_confidence","line_number","col_offset","end_col_offset","line_range","issue_cwe"]
    columns.extend(band)
    rows = []
    count=0
    last_n=500
    commits = []
    Repo = Repository(url,only_no_merge=True,order='reverse',num_workers =1,histogram_diff = True,skip_whitespaces=True)
    for x in Repo.traverse_commits():
      if (x.in_main_branch==True):
        count=count+1
        commits.append(x)
        if count == last_n:break
    in_order = []
    for value in range(len(commits)):in_order.append(commits.pop())
    commits=in_order
    for i,commit in enumerate(commits):
      print(f'[{i+1}/{len(commits)}] Mining commit {url}/commit/{commit.hash}')
      diff = []
      try:
        for m in commit.modified_files:
          if len(commit.parents) > 1:continue
          src = m.source_code_before
          dst = m.source_code
          if dst == None:continue
          f = open('temp.py','w')
          print(f'\t Mining file {m.new_path}')
          f.write(dst)
          f.close()
          os.system("bandit temp.py -f csv -o temp.csv > /dev/null 2>&1")
          df = pandas.read_csv('temp.csv')[band]
          L = df.values.tolist()
          r0 = [m.old_path,m.new_path,commit.hash,commit.parents[0],m.complexity]
          if len(L) == 0:rows.append(r0 + [None]*len(band))
          else:
            for r in L:rows.append(r0 + r)
      except:pass
    try:os.mkdir(name+'_results')
    except:pass
    f = open(name+'_results/commits_info.csv', 'w')
    f.write("")
    f.close()
    with open(name+'_results/commits_info.csv', 'a') as csvFile:
      writer = csv.writer(csvFile)
      writer.writerow(columns)
      writer.writerows(rows)
    csvFile.close()
    print('Commits info saved in '+name+'_results/commits_info.csv')

Reps = {
    "NumPy":"https://github.com/numpy/numpy",
    "OpenCV":"https://github.com/opencv/opencv-python",
    "PyTorch":"https://github.com/pytorch/pytorch",
    #"TensorFlow":"https://github.com/tensorflow/tensorflow",
    #"GeoCoder":"https://github.com/DenisCarriere/geocoder",
    #"Flask":"https://github.com/pallets/flask",
}

if __name__ == "__main__":
    for name in Reps:get_commits(Reps[name],name)