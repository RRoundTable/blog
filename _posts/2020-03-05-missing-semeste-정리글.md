---
title: "Missing Semester 정리글"
branch: master
badges: true
comments: true
categories: ['shell', 'bash']
---

# Missing Semester 정리글



## 1. Shell

리눅스의 쉘은 명령어와 프로그램을 실행할 때 사용하는 인터페이스이다.

![]({{ site.baseurl }}/images/2020-03-05-missing-semeste-정리글/shell.png "Shell")



**echo**

리눅스 명령어 echo는 주어진 문자열을, 문자열 사이에 포함된 공백과 줄 마지막에 개행문자를 포함하여 표준출력으로 출력하는 명령어다.

```
echo "Hellow Wolrd"
>>> Hellow World

echo Hellow\ Wolrd
>>> Hellow World

echo $HOME
>>> /root

echo $PATH
>>> $PATH에 포함된 주소들을 출력

which echo
>>> /usr/bin/echo # 사용하고 있는 echo 위치 출력

```

**pwd**

```
pwd
>>> /home # present working directory
```

**cd**

```
cd /roundtable # change directory
```

**. (dot) / .. (dot dot)**

```
.. # parent directory
cd .. # /home/roundtable -> /home

. # current directory
cd ./roundtable # /home -> /home/roundtable
```

**ls**

```
ls
>>> print all files in the current directory
ls -help
>>> 다양한 옵션에 대한 정보를 받아볼 수 있음
ls -l
>>> 파일에 대한 추가적인 정보를 얻을 수 있음
```

**~**

```
cd ~ # home directory
cd ~/roundtable # cd /home/roundtable
```

**cd -**

```
cd - # back to parent directory
```

**mv**

```
mv <current_file> <new_file> # rename the file or move the file in dirrent directory
```

**cp**

```
cp <current_file> <new_file> # copy
cp -r <current folder> <new_folder>
```

**rm**

```
rm <current_file> # remove
rmdir <folder>
```

**mkdir**

```
mkdir <new directory name> # make directory
```

**man**

```
man ls # manual page for ls
# if you want to quik, press q
```

**ctrl + l**: clear the terminal

**Angle bracket signs**

input stream, output stream이 존재한다. 이를 적절히 조절할 수 있다.

```
# < file 
# > file

echo hello > hello.txt # hello(print)가 hello.txt의 입력으로 들어간다.
cat hello.txt
>>> hello

cat < hello.txt > hello2.txt
cat hello2.txt
>>> hello
```

**| (pipe)**

```
# file1 | file2 # make output of file1 input of file2
# tail # 마지막 line만 출력해준다.

ls -l / tail -n1
>>> drwxrwxr-x 11 ubuntu ubuntu 4096 Mar  4 12:55 dev # print last line

curl --head --silent google.com | grep -i content-length
```

**tail**

마지막 line만 출력해준다.

**curl**

```
curl --head --silent google.com | grep -i content-length
```

**sudo: root permission**

```
sudo find -L /sys/class/backlight -maxdepth 2 -name '*brightness*'
>>> /sys/class/backlight/thinkpad_screen/brightness

cd /sys/class/backlight/thinkpad_screen

sudo echo 3 > brightness

>>> An error occurred while redirecting file 'brightness'
open: Permission denied


echo 3 | sudo tee brightness
```

**cat**: concatenate

파일의 내용을 출력

```
cat file1
# file1의 내용 출력

cat file1 file2 file3
# file1, file2, file3 이어서 출력

cat > file1 # (내용을 입력하고 ctrl + d를 눌러 저장한다.) 기존 내용을 지우고
cat >> file1 # (내용을 입력하고 ctrl + d를 눌러 저장한다.) 기존의 내용에 이어서

cat file1 file2 > file3 # file1 + file2 = file3
```

**cd /sys**

```
cd /sys # to access various kernel parameters

total 0
drwxr-xr-x   2 root root  0 Mar  5 08:03 block
drwxr-xr-x  47 root root  0 Mar  3 06:53 bus
drwxr-xr-x  69 root root  0 Mar  3 06:51 class
drwxr-xr-x   4 root root  0 Mar  5 08:03 dev
drwxr-xr-x  71 root root  0 Mar  3 04:43 devices
drwxrwxrwt   2 root root 40 Mar  3 04:43 firmware
drwxr-xr-x  12 root root  0 Mar  3 04:43 fs
drwxr-xr-x   2 root root  0 Mar  5 08:03 hypervisor
drwxr-xr-x  14 root root  0 Mar  5 08:03 kernel
drwxr-xr-x 219 root root  0 Mar  5 08:03 module
drwxr-xr-x   2 root root  0 Mar  5 08:03 power
```



shell은 단순한 argument가 아니라 일종의 프로그래밍이라고 볼 수 있다. 예를 들어서 조건문이나 반복문같은 설정을 할 수 있다.

## 2. Shell Tools and Scripting

'' 하고 "" 는 유사해보이지만, 서로 다르다.

```
foo=bar
echo "$foo"
# prints bar
echo '$foo'
# prints $foo

echo "Value is $foo"
# prints 'Value is foo'

echo 'Value is $foo'
# prints 'Value is $foo'
```

bash는  if, case, while, for와 같은 구문을 제공한다.
```
mcd (){
	mkdir -p "$1"
	cd "$1"
}
```
```
/home: vim mcd.sh
/home: source mcd.sh
/home: mcd.sh test

/home/test:  # /home -> mkdir /home/test -> cd /home/test
```

- $0 - Name of the script
- $1 to $9 - Arguments to the script. $1 is the first argument and so on.
- $@ - All the arguments
- $# - Number of arguments
- $? - Return code of the previous command
- $$ - Process Identification number for the current script
- !! - Entire last command, including arguments. A common pattern is to execute a command only for it to fail due to missing permissions, then you can quickly execute it with sudo by doing sudo !!
- $_ - Last argument from the last command. If you are in an interactive shell, you can also quickly get this value by typing Esc followed by .

```
echo "Hello"
>>> Hello

echo $?
>>> 0 # No Error

grep foobar mcd.sh
echo $?
>>> 1 # Error
```

```
# || or
false || echo "Oops, fail" 
# Oops, fail


true || echo "Will not be printed"
#

# && and
true && echo "Things went well"
# Things went well

false && echo "Will not be printed"
#

false ; echo "This will always run"
# This will always run
```


```
#!/bin/bash

echo "Starting program at $(date)" # Date will be substituted

echo "Running program $0 with $# arguments with pid $$"

for file in $@; do
    grep foobar $file > /dev/null 2> /dev/null
    # When pattern is not found, grep has exit status 1
    # We redirect STDOUT and STDERR to a null register since we do not care about them
    if [[ $? -ne 0 ]]; then
        echo "File $file does not have any foobar, adding one"
        echo "# foobar" >> "$file"
    fi
done
```

**?** : one of character
***** : any amount of characters

```

convert image.{png,jpg}
# Will expand to
convert image.png image.jpg

cp /path/to/project/{foo,bar,baz}.sh /newpath
# Will expand to
cp /path/to/project/foo.sh /path/to/project/bar.sh /path/to/project/baz.sh /newpath

# Globbing techniques can also be combined
mv *{.py,.sh} folder
# Will move all *.py and *.sh files


mkdir foo bar
# This creates files foo/a, foo/b, ... foo/h, bar/a, bar/b, ... bar/h
touch {foo,bar}/{a..j}
touch foo/x bar/y
# Show differences between files in foo and bar
diff <(ls foo) <(ls bar)
# Outputs
# < x
# ---
# > y
```

**in python**

**Shebang**은 (사전에 검색해보면) 쉬뱅이라고 읽습니다. 쉬뱅은 `#!`로 시작하는 문자열이며 스크립트의 맨 첫번째 라인에 있습니다. 쉬뱅은 유닉스 계열 운영체제에서 스크립트가 실행될 때, 파이썬, 배쉬쉘 등 어떤 인터프리터에 의해서 동작이 되는지 알려줍니다.

```
#!/usr/local/bin/python
import sys
for arg in reversed(sys.argv[1:]):
    print(arg)
```

**shellcheck**

```
$ shellcheck test.sh

In test.sh line 2:
T0=`date +%s`
   ^-- SC2006: Use $(..) instead of legacy `..`.

In test.sh line 4:
T1=`date +%s`
   ^-- SC2006: Use $(..) instead of legacy `..`.

In test.sh line 5:
ELAPSED_TIME=$((T1-T0))
^-- SC2034: ELAPSED_TIME appears unused. Verify it or export it.

In test.sh line 7:
echo "START_TIME: " ${T0}
                    ^-- SC2086: Double quote to prevent globbing and word splitting.

In test.sh line 8:
echo "END_TIME: " ${T1}
                  ^-- SC2086: Double quote to prevent globbing and word splitting.

In test.sh line 9:
echo "ELAPSED_TIME: ${ELAPSES_TIME} sec"
                    ^-- SC2153: Possible misspelling: ELAPSES_TIME may not be assigned, but ELAPSED_TIME is.
```



**export**

환경변수를 저장하는 역할, 터미널이 꺼지면 사라진다.

```
vi ~/.bashrc # 해당 주소에서 작업을 하게되면 영구적으로 남는다.

export water="삼다수"
export TEMP_DIR=/tmp
export BASE_DIR=$TEMP_DIR/backup
```

```
# gpu idx를 지정할 때 사용할 수도 있다.

export CUDA_VISIBLE_DEVICES = 1
```



**Finding how to use commands**

```
ls -h
ls --help

man ls
```

- manpage
- [TLDR pages](https://tldr.sh/): 간단하게 찾아볼 수 있음

**Finding files**

```
# Find all directories named src
find . -name src -type d
# Find all python files that have a folder named test in their path
find . -path '**/test/**/*.py' -type f
# Find all files modified in the last day
find . -mtime -1
# Find all zip files with size in range 500k to 10M
find . -size +500k -size -10M -name '*.tar.gz'

# Delete all files with .tmp extension
find . -name '*.tmp' -exec rm {} \;
# Find all PNG files and convert them to JPG
find . -name '*.png' -exec convert {} {.}.jpg \;
```



**Finding code**

[**grep**](http://man7.org/linux/man-pages/man1/grep.1.html)



```
grep foobar mcd.sh
grep -R foobar . # source code 검색도 가능
```

[**ripgrep**](https://github.com/BurntSushi/ripgrep)



```
# Find all python files where I used the requests library
rg -t py 'import requests'
# Find all files (including hidden files) without a shebang line
rg -u --files-without-match "^#!"
# Find all matches of foo and print the following 5 lines
rg foo -A 5
# Print statistics of matches (# of matched lines and files )
rg --stats PATTERN
```





**Finding shell commands**

**history**

```
history

 1 cd .\OneDrive\sourceCode\CPPS\
 2 cd ..
 3 cd .\EEN-with-Keras\

```

**Ctrl + R** : history 추적, 유용

**zsh**: 유용한 bash 도구



**Directory Naviation**

- ls -R
- [tree](https://linux.die.net/man/1/tree)
- [broot](https://github.com/Canop/broot)
- nnn
- ranger

```
ls -R
tree 
```



## 3. Git



### Git's data model

- snapshots

```
<root> (tree): snapshots, top-level directory
|
+- foo (tree)
|  |
|  + bar.txt (blob, contents = "hello world")
|
+- baz.txt (blob, contents = "git is wonderful")
```

- Modeling history: relating snapshots


```
o <-- o <-- o <-- o
^  
\
--- o <-- o
```

with the newly created merge commit shown in bold:

```
o <-- o <-- o <-- o <---- o
            ^            /
             \          v
              --- o <-- o
```

```
// a file is a bunch of bytes
type blob = array<byte>

// a directory contains named files and directories
type tree = map<string, tree | file>

// a commit has parents, metadata, and the top-level tree
type commit = struct {
    parent: array<commit>
    author: string
    message: string
    snapshot: tree
}

type object = blob | tree | commit # 모두 다 object다


objects = map<string, object>

def store(object):
    id = sha1(object)
    objects[id] = object

def load(id):
    return objects[id]
```

**References**: HEAD

```
references = map<string, string>

def update_reference(name, id):
    references[name] = id

def read_reference(name):
    return references[name]

def load_reference(name_or_id):
    if name_or_id in references:
        return load(references[name_or_id])
    else:
        return load(name_or_id)
```



### Hook

특정상황에서 특정 스크립트를 실행할 수 있도록 하는 기능

위치: cd ./git/hook

```
#!/bin/sh

# git diff --exit-code --cached --name-only --diff-filter=ACM -- '*.png' '*.jpg'
# 위의 명령어는 현재 add 되어있는 파일 중, .png와 .jpg 확장자를 가진 파일들을 '이름만' 추출합니다.
images=$(git diff --exit-code --cached --name-only --diff-filter=ACM -- '*.png' '*.jpg')

# 추출된 이미지 파일들을 ImageOptimCLI에 넘겨주기만 하면 되는 것이죠!
# 이미지들이 압축되어 변경되었으니 다시 add 해줘야겠죠?
$(exit $?) || (echo "$images" | ~/.woowa/imageoptim-cli/bin/imageOptim && git add $images)

```

- **pre-commit**: https://pre-commit.com/

- reference: https://woowabros.github.io/tools/2017/07/12/git_hook.html



### Github Deployment & Actions

https://blog.banksalad.com/tech/become-an-organization-that-deploys-1000-times-a-day/?fbclid=IwAR1X6CC1mz6Akrxcyt-BpeMZ-ZnpLOvdGlK7dvxh0De85D1qsEoLN2JEhAU

### Git command-line interface

```
git help <command>: get help for a git command
git init: creates a new git repo, with data stored in the .git directory
git status: tells you what’s going on
git add <filename>: adds files to staging area
git commit: creates a new commit
git log: shows a flattened log of history
git log --all --graph --decorate: visualizes history as a DAG
git diff <filename>: show differences since the last commit
git diff <revision> <filename>: shows differences in a file between snapshots
git checkout <revision>
```



**Branching and merging**

```
git branch: shows branches
git branch <name>: creates a branch
git checkout -b <name>: creates a branch and switches to it
same as git branch <name>; git checkout <name>
git merge <revision>: merges into current branch
git mergetool: use a fancy tool to help resolve merge conflicts
git rebase: rebase set of patches onto a new base
```



**Remotes**

```
git remote: list remotes
git remote add <name> <url>: add a remote
git push <remote> <local branch>:<remote branch>: send objects to remote, and update remote reference
git branch --set-upstream-to=<remote>/<remote branch>: set up correspondence between local and remote branch
git fetch: retrieve objects/references from a remote
git pull: same as git fetch; git merge
git clone: download repository from remote
```



**Undo**

```
git commit --amend: edit a commit’s contents/message
git reset HEAD <file>: unstage a file
git checkout -- <file>: discard changes
```
- https://git-scm.com/book/en/v2


### Github로 협업하기

**checkout**
https://mytory.net/archives/10078

**stash**


특정 커밋 선택해서 반영하기:cherry pick

```
git cherry-pick {Commit ID}
```

여러개의 커밋을 반영하기: rebase

```
git rebase {가져올 Branch 이름}
```

- https://www.tuwlab.com/ece/22218



### **Github Actions**

- Build

  코드 스타일 검사를 위한 lint

  유닛 테스트를 실행하는 test

  docker image를 build하는 build

- Deploy

  아직은 경험하기 힘든 영역이라고 생각



### Gitflow

- https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow



### Github slack integration

- https://slack.github.com/

- http://wepla.net/?p=2353



### Github Commit message

```
Type: 제목

본문

꼬리말
```



feat: 새로운 기능을 추가할 경우
fix: 버그를 고친 경우
docs: 문서 수정한 경우
style: 코드 포맷 변경, 세미 콜론 누락, 코드 수정이 없는 경우
refactor: 프로덕션 코드 리팩터링
test: 테스트 추가, 테스트 리팩터링 (프로덕션 코드 변경 없음)
chore: 빌드 테스크 업데이트, 패키지 매니저 설정할 경우 (프로덕션 코드 변경 없음)

```
feat: Summarize changes in around 50 characters or less

More detailed explanatory text, if necessary. Wrap it to about 72
characters or so. In some contexts, the first line is treated as the
subject of the commit and the rest of the text as the body. The
blank line separating the summary from the body is critical (unless
you omit the body entirely); various tools like `log`, `shortlog`
and `rebase` can get confused if you run the two together.

Explain the problem that this commit is solving. Focus on why you
are making this change as opposed to how (the code explains that).
Are there side effects or other unintuitive consequenses of this
change? Here's the place to explain them.

Further paragraphs come after blank lines.

 - Bullet points are okay, too

 - Typically a hyphen or asterisk is used for the bullet, preceded
   by a single space, with blank lines in between, but conventions
   vary here

If you use an issue tracker, put references to them at the bottom,
like this:

Resolves: #123
See also: #456, #789
```



- https://sujinlee.me/professional-github/
- https://junwoo45.github.io/2020-02-06-commit_template/?fbclid=IwAR2HKgwO9imOxWAvWUPtaXDymUzMRRJ18LnwR_Cwa3s6kcrFidIwvz8CvmY

## 4. [Python CI: 핑퐁팀 사례](https://blog.pingpong.us/python-in-pingpong/?fbclid=IwAR0ijQfAoKTFtXsppYN-b9iqw_vnnR6VoEMcL9Wr8TJFUyHnuOBwBgyHCZw#%EC%8B%9C%EC%9E%91%EC%9D%80-%EA%B0%80%EB%B3%8D%EA%B2%8C-%EC%BD%94%EB%93%9C-%ED%92%88%EC%A7%88-%EA%B4%80%EB%A6%AC-%EB%8F%84%EA%B5%AC)

**코드 스타일 확인 및 포맷 자동화****

- [black](https://github.com/psf/black)
	Python Software Foundation에서 작성한 Python 자동 포맷팅 도구입니다. pycodestyle을 따른다.
	
- flake8
	
	**pycodestyle + pyflakes + 복잡도 검사 기능**
	
	Python Code Quality Authority (PyCQA)에서 작성한 스타일 체크 도구로, 플러그인을 붙이기 쉬운 것이 장점이다. docstring 형식 부분을 잡아내기 위해서 적용하기도 한다. 
	
- yapf
	구글에서 배포하는 자동 포맷팅 도구입니다. 다른 포맷팅 도구들이 스타일 가이드를 어긴 부분만 잡아준다면, yapf는 스타일 가이드를 어기지 않았더라도 다시 포맷팅을 진행하는 상당히 엄격한 자동 포맷팅 도구입니다.
	
- isort: import statement 정렬하는 도구



**Type Checker**

Type Hints란, 예상치 못한 타입이 변수에 할당되는 것을 검사기가 막아주는 역할을 한다. 

- [pyright](https://github.com/microsoft/pyright): vscode와 연동가능
- mypy
- pyre-check



**CI**

- CircleCI
- Jenkins Blueocean
- GitHub Actions
- travisCI



**테스트 코드**

- [pytest](https://docs.pytest.org/en/latest/)
- unittest



**Code Coverage**

- pytest-cov



**초기 템플릿 생성**





## 5. Debugging and Profiling



### Logging

- print문 삽입하는 법
- [log](https://docs.python.org/ko/3/howto/logging.html): 일반적으로 log가 더 좋은 방법



**Log의 장점**

1. log를 사용하면 file로 저장할 수 있다. (remote server에도 가능)
2. Serverity level별로 나타낼 수 있다.
   1. INFO
   2. DEBUG
   3. WARN
   4. ERROR
3. 새로운 이슈가 추가될 때, 더 골고루 살펴볼 수 있다.



python logging 모듈 활용하기

```python

import logging

# logger instance
logger = logging.getLogger(__name__)

# handler
streamHandler = logging.StreamHandler()
fileHandler = logging.FileHandler('./server.log')


logger.addHandler(streamHandler)
logger.addHandler(FileHandler)


logger.setLevel(level=logging.DEBUG)
logger.debug('원하는 log 문 작성하기')

```

- https://hamait.tistory.com/880



### Debugger

[pdb](https://docs.python.org/3/library/pdb.html)

### Static Analysis

[shellcheck](https://www.shellcheck.net/)

### linting

잠재적인 오류에 대한 코드를 분석하는 프로그램.

flake8을 이용하여 진행할 수 있다.


### Profiling


#### Timing

```python
import time, random

n = random.randint(1, 10) * 100

# Get current time
start = time.time()

# Do some work
print("Sleeping for {} ms".format(n))
time.sleep(n/1000)

# Compute time between start and now
print(time.time() - start)

# Output
# Sleeping for 500 ms
# 0.5713930130004883
```

위의 예시처럼 시간을 측정하면, 실제 시간과 차이가 나는 경우가 있다. 예를 들면, 다른 작업이 cpu를 할당 받고 있어서 그 후에 실행된 경우가 그러한 경우이다. 따라서, 해당 소스코드의 동작시간을 알고 싶다면, User가 사용한 시간 + system이 사용한 시간을 더해서 구할 수 있다. (User + Sys)

- Real - Wall clock elapsed time from start to finish of the program, including the time taken by other processes and time taken while blocked (e.g. waiting for I/O or network)
- User - Amount of time spent in the CPU running user code
- Sys - Amount of time spent in the CPU running kernel code


#### Profilers
##### CPU

tracing, sampling profilers 두 가지의 종류가 있다.

tracing profiler는 모든 function call을 기록하는 반면에 sampling profiler는 특정 간격마다 기록한다.

python의 경우 cProfile 모듈을 이용할 수 있다. 아래와 같은 소스코드가 있다고 가정하자.

```python
#!/usr/bin/env python

import sys, re

def grep(pattern, file):
    with open(file, 'r') as f:
        print(file)
        for i, line in enumerate(f.readlines()):
            pattern = re.compile(pattern)
            match = pattern.search(line)
            if match is not None:
                print("{}: {}".format(i, line), end="")

if __name__ == '__main__':
    times = int(sys.argv[1])
    pattern = sys.argv[2]
    for i in range(times):
        for file in sys.argv[3:]:
            grep(pattern, file)

```
아래와 같이 프로파일링을 진행할 수 있다. 모든 function call을 확인할 수 있다.
```
$ python -m cProfile -s tottime grep.py 1000 '^(import|\s*def)[^,]*$' *.py

[omitted program output]

 ncalls  tottime  percall  cumtime  percall filename:lineno(function)
     8000    0.266    0.000    0.292    0.000 {built-in method io.open}
     8000    0.153    0.000    0.894    0.000 grep.py:5(grep)
    17000    0.101    0.000    0.101    0.000 {built-in method builtins.print}
     8000    0.100    0.000    0.129    0.000 {method 'readlines' of '_io._IOBase' objects}
    93000    0.097    0.000    0.111    0.000 re.py:286(_compile)
    93000    0.069    0.000    0.069    0.000 {method 'search' of '_sre.SRE_Pattern' objects}
    93000    0.030    0.000    0.141    0.000 re.py:231(compile)
    17000    0.019    0.000    0.029    0.000 codecs.py:318(decode)
        1    0.017    0.017    0.911    0.911 grep.py:3(<module>)

[omitted lines]
```

line마다 profiling 하고 싶다면, kernprof를 사용할 수 있다. 단, 데코레이터를 사용해야한다.

```
#!/usr/bin/env python
import requests
from bs4 import BeautifulSoup

# This is a decorator that tells line_profiler
# that we want to analyze this function
@profile
def get_urls():
    response = requests.get('https://missing.csail.mit.edu')
    s = BeautifulSoup(response.content, 'lxml')
    urls = []
    for url in s.find_all('a'):
        urls.append(url['href'])

if __name__ == '__main__':
    get_urls()
```

```
$ kernprof -l -v a.py
Wrote profile results to urls.py.lprof
Timer unit: 1e-06 s

Total time: 0.636188 s
File: a.py
Function: get_urls at line 5

Line #  Hits         Time  Per Hit   % Time  Line Contents
==============================================================
 5                                           @profile
 6                                           def get_urls():
 7         1     613909.0 613909.0     96.5      response = requests.get('https://missing.csail.mit.edu')
 8         1      21559.0  21559.0      3.4      s = BeautifulSoup(response.content, 'lxml')
 9         1          2.0      2.0      0.0      urls = []
10        25        685.0     27.4      0.1      for url in s.find_all('a'):
11        24         33.0      1.4      0.0          urls.append(url['href'])
```

##### Memory

python을 사용할 시에, 데코레이터를 사용한 후, memory_profiler를 사용하면, 메모리 사용량을 검사할 수 있다.

```python
@profile
def my_func():
    a = [1] * (10 ** 6)
    b = [2] * (2 * 10 ** 7)
    del b
    return a

if __name__ == '__main__':
    my_func()

```

```
$ python -m memory_profiler example.py
Line #    Mem usage  Increment   Line Contents
==============================================
     3                           @profile
     4      5.97 MB    0.00 MB   def my_func():
     5     13.61 MB    7.64 MB       a = [1] * (10 ** 6)
     6    166.20 MB  152.59 MB       b = [2] * (2 * 10 ** 7)
     7     13.61 MB -152.59 MB       del b
     8     13.61 MB    0.00 MB       return a
```



> 파이썬에서 데코레이터란
>
> https://wikidocs.net/23106





##### Event Profiling

- strace
- perf

#### Visualization

- Call graphs: python - [pycallgraph](http://pycallgraph.slowchop.com/en/master/)
- Flame Graph



### Resource Monitoring

- **General Monitoring** - Probably the most popular is [`htop`](https://hisham.hm/htop/index.php), which is an improved version of [`top`](http://man7.org/linux/man-pages/man1/top.1.html). `htop` presents various statistics for the currently running processes on the system. `htop` has a myriad of options and keybinds, some useful ones are: `` to sort processes, `t` to show tree hierarchy and `h` to toggle threads. See also [`glances`](https://nicolargo.github.io/glances/) for similar implementation with a great UI. For getting aggregate measures across all processes, [`dstat`](http://dag.wiee.rs/home-made/dstat/) is another nifty tool that computes real-time resource metrics for lots of different subsystems like I/O, networking, CPU utilization, context switches, &c.
- **I/O operations** - [`iotop`](http://man7.org/linux/man-pages/man8/iotop.8.html) displays live I/O usage information and is handy to check if a process is doing heavy I/O disk operations
- **Disk Usage** - [`df`](http://man7.org/linux/man-pages/man1/df.1.html) displays metrics per partitions and [`du`](http://man7.org/linux/man-pages/man1/du.1.html) displays **d**isk **u**sage per file for the current directory. In these tools the `-h` flag tells the program to print with **h**uman readable format. A more interactive version of `du` is [`ncdu`](https://dev.yorhel.nl/ncdu) which lets you navigate folders and delete files and folders as you navigate.
- **Memory Usage** - [`free`](http://man7.org/linux/man-pages/man1/free.1.html) displays the total amount of free and used memory in the system. Memory is also displayed in tools like `htop`.
- **Open Files** - [`lsof`](http://man7.org/linux/man-pages/man8/lsof.8.html) lists file information about files opened by processes. It can be quite useful for checking which process has opened a specific file.
- **Network Connections and Config** - [`ss`](http://man7.org/linux/man-pages/man8/ss.8.html) lets you monitor incoming and outgoing network packets statistics as well as interface statistics. A common use case of `ss` is figuring out what process is using a given port in a machine. For displaying routing, network devices and interfaces you can use [`ip`](http://man7.org/linux/man-pages/man8/ip.8.html). Note that `netstat` and `ifconfig` have been deprecated in favor of the former tools respectively.
- **Network Usage** - [`nethogs`](https://github.com/raboof/nethogs) and [`iftop`](http://www.ex-parrot.com/pdw/iftop/) are good interactive CLI tools for monitoring network usage.



## 6. Metaprogramming

- paper
- source code
- tool
- dependency

### Build systems

**make** 

ubuntu에서 make 명령어는 파일 관리 유틸리티이다. make 명령어를 실행하는 위치에 Makefile이 있어야 한다.

Makefile은 다음과 같은 구조를 가진다.

- Target: 명령어가 실행되어 나온 결과를 저장할 파일
- Dependency: Target을 만들기 위해 필요한 재료
- Command: 실행되어야 할 명령어들
- macro: 코드를 단순화 시키기 위한 방법

```
CC=<컴파일러>
CFLAGS=<컴파일 옵션>
LDFLAGS=<링크 옵션>
LDLIBS=<링크 라이브러리 목록>
OBJS=<Object 파일 목록>
TARGET=<빌드 대상 이름>
 
all: $(TARGET)
 
clean:
    rm -f *.o
    rm -f $(TARGET)
 
$(TARGET): $(OBJS)
$(CC) -o $@ $(OBJS)
```

빌드규칙 블록

```
<Target>: <Dependencies>
    <Recipe>
```

- **Target:** 빌드 대상 이름. 통상 이 Rule에서 최종적으로 생성해내는 파일명을 써 줍니다.
- **Dependencies:** 빌드 대상이 의존하는 Target이나 파일 목록. 여기에 나열된 대상들을 먼저 만들고 빌드 대상을 생성합니다.
- **Recipe:** 빌드 대상을 생성하는 명령. 여러 줄로 작성할 수 있으며, 각 줄 시작에 반드시 Tab문자로 된 Indent가 있어야 합니다.



예시

```
paper.pdf: paper.tex plot-data.png
	pdflatex paper.tex

plot-%.png: %.dat plot.py
	./plot.py -i $*.dat -o $@

# %: pattern
# $@: current target
# reference: http://www.gnu.org/software/make/manual/html_node/Automatic-Variables.html
```

- 자세한 정보를 알고 싶다면, https://www.tuwlab.com/ece/27193



### Dependency management

[semantic versioning](https://spoqa.github.io/2012/12/18/semantic-versioning.html)


### Continuous integration systems

- Github Actions
- Travis CI
- Pipelines


### A brief aside on testing
- test suite
- unit test
- intergration test
- regression test
- Mocking


- Reference: https://missing.csail.mit.edu/2020/?fbclid=IwAR2gQe5LToKuqVUwbfegqSOk6BnIqscbnqjK0e3js64EceMswNqW0KgeSEo