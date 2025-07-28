AI

### 安装依赖
`python -m pip install --upgrade pip setuptools wheel`
`pip install -r requirements.txt`

### 生成依赖文件
`pip list --format=freeze > requirements.txt`

### 以上方式可以锁定版本，目前不要，自己开发学习后面不要锁定版本，跟上潮流
`pip install -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple -r .\requirements.txt`


使用pytorch同时使用vllm部署
```bash
$ conda create -n vllm python=3.12
$ python -m pip install --upgrade pip setuptools wheel
$ pip install uv
$ uv pip install vllm --torch-backend=auto
```