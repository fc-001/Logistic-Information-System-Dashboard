# Project for Logistic Information System: A Dashboard
# 物流信息系统大作业

### 使用方法：
1. 先配好新建一个环境(本实验是Python3.10)`conda create -n your_env_name python==3.10`;
2. 在终端运行`conda activate your_env_name`;
3. 在终端运行`pip install -r requirements.txt`;
4. 安装好所有依赖后继续运行`python dashboard.py`;
5. 等终端跑完两轮代码后，可以在浏览器打开对应的端口，利用自身的ip加上:8001即可(自身主机ip的获取方式：在终端输入`ipconfig`(Windows)，Mac或者Linux则是`ifconfig`后回车，找到"inet"后面的xxx.x.x.x)。
比如我的电脑是127.0.0.1，则在浏览器输入"127.0.0.1:8001"即可打开仪表盘;
6. 若要更改后再次运行，请先在终端Ctrl+C中断进程后进行更改，运行时再次`python dashboard.py`即可;
7. 若只要看部分图的，请参考Notebook文件夹.

### Instructions:
1. Create a new environment(Python3.10)
2. Run in terminal`conda activate your_env_name`
3. Run in terminal`pip install -r requirements.txt`
4. Run in terminal`python dashboard.py`
5. Wait about 30 seconds until the terminal is quiet, then open the dashboard in your browser. First you need to get the IP of your device. Run `ipconfig` for Windows or `ifconfig` for Mac/Linux in terminal.
Find the string of numbers after "inet". For example, mine is 127.0.0.1, then add ":8001" behind and open in the browser.
6. If a change is needed, you should first press Ctrl+C in terminal, then revise the code until it meets your needs, then re-run `python dashboard.py` in terminal.
7. Only see single pictures? Please refer to the files in `Notebook`.


数据集`data.xlsx`。我们用`data_filter.py`进行了一些预处理，其他部分都写在Jupyter Notebook里。

由于主要是做仪表盘进行可视化，因此我们注重功能的实现而非高超的算法和数据处理技术。首先在`hub2hub.ipynb`中，我们主要针对路线进行可视化，分为总仓到分仓，分仓之间和分仓到门店。
其次是`operations_metrics.ipynb`里面，主要是发运量上的可视化，展现了一些运量、成本方面的指标。而最后的`inventory.ipynb`则是更多关注了库存管理的数据，我们假设期初库存为0进行了所有的可视化。EOQ和安全库存的公式可以在《供应链管理》的相关教科书或者网络资源中找到。

对应的$\textbf{Python}$文件是由$\textbf{Notebook}$转换而来的，为了构建仪表盘。
### 仅供参考！



This is a project example for the course: Logistic Information System at School of Transportation, Tongji University.

The origin dataset `data.xlsx` is given by the lecturer, Prof. Ning Tian at Tongji University.

The requirement is to design a dashboard that can effectively illustrate the sales data, which may provide some suggestions for company's operations management.

All is coded in Python and shown in those $\textbf{Jupyter Notebooks}$, besides the data filtering process.

In `hub2hub.ipynb`, we design some figures for routes illustration between NDC_Changshu to RDCs, and RDCs to other RDCs, as well as RDCs to stores.

In `operations_metrics.ipynb`, we provide transport volumes from different perspectives.

In `inventory.ipynb`, we make some efforts in demonstrate the condition of inventory that should be serious considered in realistic management.
The calculation methods and formulas can be found in any textbooks or sources that introduce $\textbf{Supply Chain Management}$.

Should you have any questions or suggestions, feel free to contact: 2251140@tongji.edu.cn, or fancheng2251140@gmail.com

### Just for your reference!




