# Introduction

This lab explores automated testing techniques to improve the quality and reliability of Python software. It leverages a suite of industry-standard tools and frameworks, each playing a crucial role in the testing process:

- **PyTest**  
  A robust testing framework that simplifies writing and executing tests for Python projects. Its flexibility and extensive ecosystem—including plugins like pytest-cov and pytest-func-cov—make it an essential component for continuous integration and quality assurance workflows. [PyTest Documentation](https://docs.pytest.org/en/7.4.x/index.html)

- **Coverage**  
  A tool that measures which parts of the code are executed during tests, helping developers identify untested sections. Integrated with PyTest via plugins like pytest-cov, it provides detailed insights into test effectiveness. [Coverage Documentation](https://coverage.readthedocs.io/en/latest)

- **Pynguin**  
  An automated test case generation tool that creates test cases aiming to increase code coverage. While Pynguin can identify and exercise more code paths—raising coverage metrics—it often requires manual refinement to validate correct behavior. [Pynguin Website](https://www.pynguin.eu)

- **Additional Plugins: pytest-cov & pytest-func-cov**  
  These plugins seamlessly integrate coverage analysis into the PyTest workflow, providing developers with immediate feedback on test coverage during routine test execution. [pytest-cov](https://github.com/pytest-dev/pytest-cov) | [pytest-func-cov](https://pypi.org/project/pytest-func-cov)

Testing can broadly be divided into static and dynamic approaches, each with distinct characteristics:

- Static Testing: 
  - Sound but Imprecise: It aims to be conservative by over-approximating potential issues, which can result in imprecision.  
  - Input-Oblivious: Static techniques analyze code without executing it, meaning they do not account for runtime input variations.

- Dynamic Testing:
  - Incomplete but Precise: It may not cover every possible execution path, as it relies on specific input data to drive the tests.  
  - Input-Dependent: By executing the code, dynamic testing provides precise feedback for the specific scenarios tested.

The purpose of this lab is to explore dynamic testing techniques, focusing on improving code coverage in a Python project. The lab demonstrates how PyTest and Coverage can be used to measure existing test effectiveness, followed by an experiment using Pynguin to automatically generate additional test cases. 

The lab ultimately reinforces the idea that automated test generation can be a useful supplement to human-written tests, improving coverage and potentially detecting edge cases. However, manual verification remains essential to ensure the quality and correctness of the generated test cases.




# Methodology and Execution

- Cloning the repository
    
    ![image.png](Lab%205%201925a341429b80ccbe40eaf5ab90027c/image.png)
    
    Since the latest commit didn’t have a green tick, I reverted back to the last one which did, which has hash `1117ffe` .
    
    ![image.png](Lab%205%201925a341429b80ccbe40eaf5ab90027c/image%201.png)
    
- testing using `pytest`
    
    ![image.png](Lab%205%201925a341429b80ccbe40eaf5ab90027c/image%202.png)
    
- Getting line coverage using `coverage`
    
    First I ran :
    
    ```
    coverage -m pytest tests
    ```
    
    This gave me the same output as simply running `pytest` , but allowed line coverage to be recorded too.
    Then, I created a HTML visualisation like this :
    
    ![image.png](Lab%205%201925a341429b80ccbe40eaf5ab90027c/image%203.png)
    
    ![image.png](Lab%205%201925a341429b80ccbe40eaf5ab90027c/image%204.png)
    
    ![image.png](Lab%205%201925a341429b80ccbe40eaf5ab90027c/image%205.png)
    
    Thus, there is only 90% code coverage.
    
    We want more than that, and so we will generate another test case suite using `pynguin`.
    
    We first get the files with uncovered lines and store it in a text file
    
    ![image.png](Lab%205%201925a341429b80ccbe40eaf5ab90027c/image%206.png)
    
    Then, with the help of a simple python script, we change this to a usable format, like this :
    
    ![image.png](Lab%205%201925a341429b80ccbe40eaf5ab90027c/image%207.png)
    
- Generating test cases
    
    Then, we programmatically run `pynguin` for all these files using this script :
    
    ```python
    import os
    from multiprocessing import Process, Queue
    import time
    L = open("test_A_uncov_clean.txt",'r').read().split("\n")
    dt = 5
    T = 120
    def run_pynguin(x):
        y =  x.split("/")
        project = "/".join(y[:-1])
        module = y[-1][:-3]
        command = f"pynguin --project-path {project} --output-path pynguin-results --module-name {module} --maximum-search-time 60 --maximum-test-execution-timeout 60"
        print(command)
        os.system(command)
    for x in L:
        kill = False
        finished = False
        queue = Queue()
        p = Process(target=run_pynguin,args=(x,))
        p.start()
        t0 = time.time()
        while not kill and not finished:
            time.sleep(dt)
            t = time.time()
            t = t - t0
            if not p.is_alive():
                finished = True
            elif t > T:
                print(f"killed pynguing for {x}")
                kill = True
        if kill:
            while p.is_alive():
                os.system(f"kill -9 {p.pid}")
        else:
            p.join(10)
            #RETURN_VALS = queue.get(timeout=10)
    
    ```
    
    The output thus generated is
    
    1. The test cases in `pynguin-results/` (which, we will call test suit B) :
        
        ![image.png](Lab%205%201925a341429b80ccbe40eaf5ab90027c/image%208.png)
        
    2. The code coverage of the files using the generated test-cases :
        
        ![image.png](Lab%205%201925a341429b80ccbe40eaf5ab90027c/image%209.png)
        
    
    We can use the `statistics.csv` and the `test_A_uncov.txt` files to analyse how the generated test cases differ from the old ones. I’m using this script for the analysis:
    
    ```python
    import matplotlib.pyplot as plt
    import pandas as pd
    data = pd.read_csv('test_A_uncov_clean.csv').drop_duplicates(subset='TargetModule')
    new_data = pd.read_csv('pynguin-report/statistics.csv').drop_duplicates(subset='TargetModule')
    new_data["Coverage"] = new_data["Coverage"].astype('float')
    data["Coverage"] = data["Coverage"].astype('float') / 100
    data.index = data["TargetModule"]
    new_data.index = new_data["TargetModule"]
    data = data.drop(columns=["TargetModule"])
    new_data = new_data.drop(columns=["TargetModule"])
    data = data.sort_index()
    new_data = new_data.sort_index()
    data = data.join(new_data, rsuffix="_new")
    print("Old average coverage:",data["Coverage"].mean())
    print("New average coverage:",data["Coverage_new"].mean())
    data.plot.scatter(x='Coverage', y='Coverage_new')
    plt.xlabel("Old Coverage")
    plt.ylabel("New Coverage")
    plt.savefig("Coverage.png")
    ```
    
    Output:
    
    ```python
    Old average coverage: 0.7631746031746034
    New average coverage: 0.817315261873147
    ```
    
    ![Coverage.png](Lab%205%201925a341429b80ccbe40eaf5ab90027c/Coverage.png)
    
    Note that this output is only for the files where code coverage was already low.
    
    It shows that `pynguin` increases code coverage only slightly.
    
- Running `pytest` on test suite B
    
    The directory structure makes our generated tests unusable with `pytest` . So, using this script, I merged the `algorithms` and `pynguin-results` directories into `test_suit_B` :
    
    ```python
    import os
    folder = "test_suit_B"
    def pull_up(folder):
        for subfolder in os.listdir(folder):
            if subfolder[-3:] == ".py":continue
            if subfolder == "__pycache__":continue
            #if subfolder == "pynguin-results":continue
            sf = folder + "/" + subfolder
            pull_up(sf)
            try:L = os.listdir(sf)
            except:continue
            for x in L:
                if x == "__init__.py":continue
                if x[-3:] != ".py":continue
                file = sf + "/" + x
                upper = folder + "/" + x
                print(file)
                file = open(file, "r")
                s = file.read()
                file.close()
                try:
                    upper = open(upper, "a")
                    upper.write("\n\n\n")
                except:upper = open(upper, "w")
                upper.write(s)
                upper.close()
    pull_up(folder)
    ```
    
    Then, the output of running `pytest` was:
    
    ![image.png](Lab%205%201925a341429b80ccbe40eaf5ab90027c/image%2010.png)
    
    So, the tests, although do cover a bit more code are mostly wrong, since there is a massive difference between the outputs of test suite A and B upon running `pytest` .

# Results

There is only 90% code coverage in the actual test suite for `keon/algorithms` . 

After collecting all the units with low covered line, we get a coverage of 76.3 % on these units. 

When test cases are generated using `pynguin`, this improves to 81.7 % ,that is, there is some marginal improvement. 

This improvement is mostly due to increased coverage in modules with extremely low coverage, as seen in this plot :

![Coverage.png](Lab%205%201925a341429b80ccbe40eaf5ab90027c/Coverage.png)

Of course, the "tests" generated by `pynguin` are nothing but inputs that allow good line coverage, where the expected output in the test case isn't the correct output in the majority of generated test cases. So, a human must intervene and "fix" these tests. This, will lead to somewhat better code coverage in the end and is a time-effective way that doesn't rely on developers manually writing unit tests.

# Conclusion

PyTest, combined with `coverage`, is an invaluable toolset for testing Python modules and analyzing code coverage. These tools streamline the testing process, making it easier to identify untested portions and improve software reliability.

Automating test case generation with tools like Pynguin can further enhance efficiency. While Pynguin increases code coverage, as shown by the improvement from 76.3% to 81.7%, its generated test cases often require manual corrections. These tests execute more code but do not always validate correct program behavior, making human oversight essential.

A balanced approach is ideal: Pynguin can generate an initial batch of test cases, which developers then refine to ensure correctness. This hybrid method leverages automation for efficiency while maintaining accuracy, leading to more reliable and maintainable test suites.

