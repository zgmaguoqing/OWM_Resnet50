# OWM_Resnet50

This repository achieved **7nd Place Award​​** in the The 2nd International Algorithm and Computing Power Competition 2023, hosted by Pengcheng Laboratory - Continual Learning Track.

<img width="672" height="378" alt="competition_track_prize" src="https://github.com/user-attachments/assets/cf057f98-23fc-4718-88f4-378579d87caf" />


## Dataset
<!-- 创建`data`文件夹,与`OWM_Resnet50`文件夹并列。将程序中`shift.CIFAR10`下载的文件放在`data`目录下即可。
- 加载一次数据集后，程序会将处理后的数据集对象保存。保存路径在`data`文件夹下，文件名为`cida10.dataset`-->

- Create a `data` folder at the same level as the `OWM_Resnet50` folder.
  
- Place the files downloaded by `shift.CIFAR10` into the `data` directory.
  
- After loading the dataset once, the program will save the processed dataset object. The save path is in the `data` folder, with the file named `cifa10.dataset`.
   
- In the 10split, each task has 5,000 images, with 10 tasks, resulting in a training set of 50,000 images in total. The test set has 500 images per task, with 10 tasks, resulting in a test set of 5,000 images in total.

- In the 4split, each task has 5,000 images, with 10 tasks, resulting in a training set of 50,000 images in total. The test set has 500 images per task, with 10 tasks, resulting in a test set of 5,000 images in total.

## Problem Setting

<img width="866" height="295" alt="连续学习范式" src="https://github.com/user-attachments/assets/7ce4a584-3070-4706-890e-9f4ce2230e7b" />

## Run the code
镜像说明：使用评测专用镜像，python3.9.16，依赖包见Dockerfile和requirements.txt。cuda版本为11.7、torch2.0、numpy1.25.2时运行无错误

预计的训练时间:10splitTasks一个任务训练时间约为1小时30分钟，共约12小时。4splitDomains整个序列任务训练时间约为3小时。

其他细节说明: 训练时请在config.py中选择
                                    _C.AGENT.TYPE = 'trainer_OWM'
                                    _C.AGENT.NAME = 'OWMTrainer'

特殊说明： 

1. Q&A里说不允许修改CLTrainer.learn_task()方法，代码里有修改，但并未违反规则。连续学习策略为改进的OWM，使用训练完第一个任务的特征提取器，在后续任务上仅通过OWM和AOP正交修改的方式微调resnet50的最后三个卷积层和分类头。

######################################################
4splitDomains # 训练（复现）
Task 0
```
python iBatchLearn.py --cfg ./4splitDomains.yaml --user_cfg ./utils/user_4splitDomains.yaml --task_count 0 \
--suffix local_test --init_path ./init_model/4splitDomains.pth --ckpt_path ./init_model/4splitDomains.pth --save_ckpt_path ./model_info/4splitDomains/checkpoint-0.pth \
--save_storage_path ./model_info/4splitDomains/storage-0.pth
```

Task 1 其余任务同理
```
python iBatchLearn.py --cfg ./4splitDomains.yaml --user_cfg ./utils/user_4splitDomains.yaml --task_count 1 \
--suffix local_test --init_path ./init_model/4splitDomains.pth --ckpt_path ./model_info/4splitDomains/checkpoint-0.pth --save_ckpt_path ./model_info/4splitDomains/checkpoint-1.pth \
--storage_path ./model_info/4splitDomains/storage-0.pth --save_storage_path ./model_info/4splitDomains/storage-1.pth
```

4splitDomains # 推理（评审）
```
python iBatchLearn.py --cfg ./4splitDomains.yaml --user_cfg ./utils/user_4splitDomains.yaml --test --task_count 0 \
--suffix local_test --init_path ./init_model/4splitDomains.pth --ckpt_path ./model_info/4splitDomains/checkpoint-0.pth  \
--dest_path ./dest-4splitDomains.pkl
```

######################################################
10splitTasks # 训练（复现）
Task 0
```
python iBatchLearn.py --cfg ./10splitTasks.yaml --user_cfg ./utils/user_10splitTasks.yaml --task_count 0 \
--suffix local_test --init_path ./init_model/10splitTasks.pth --ckpt_path ./init_model/10splitTasks.pth --save_ckpt_path ./model_info/10splitTasks/checkpoint-0.pth \
--save_storage_path ./model_info/10splitTasks/storage-0.pth
```

Task 1 其余任务同理
```
python iBatchLearn.py --cfg ./10splitTasks.yaml --user_cfg ./utils/user_10splitTasks.yaml --task_count 1 \
--suffix local_test --init_path ./init_model/10splitTasks.pth --ckpt_path ./model_info/10splitTasks/checkpoint-0.pth --save_ckpt_path ./model_info/10splitTasks/checkpoint-1.pth \
--storage_path ./model_info/10splitTasks/storage-0.pth --save_storage_path ./model_info/10splitTasks/storage-1.pth
```

10splitTasks # 推理（评审）
```
python iBatchLearn.py --cfg ./10splitTasks.yaml --user_cfg ./utils/user_10splitTasks.yaml --test --task_count 0 \
--suffix local_test --init_path ./init_model/10splitTasks.pth --ckpt_path ./model_info/10splitTasks/checkpoint-0.pth  \
--dest_path ./dest-10splitTasks.pkl
```

## Acknowledgments


主办单位

鹏城实验室

协办单位

中国工业与应用数学学会（大数据与人工智能专业委员会）

中国计算机学会

中国指挥与控制学会

中国人工智能学会

工业和信息化部电子第五研究所

西安电子科技大学广州研究院

国家超算互联网联合体

华为技术有限公司

陕西长安先导产业创新中心有限公司

西安人才集团有限公司

大赛指导委员会

袁亚湘（中国科学院院士、中科院数学院研究员）

戴琼海（中国工程院院士、中国人工智能学会理事长、清华大学教授）

梅　宏（中国科学院院士、发展中国家科学院院士、北京大学教授）

王怀民（中国科学院院士、国防科技大学教授）

戴　浩（中国工程院院士、军事科学院系统工程研究院研究员）

徐宗本（中国科学院院士、西安交通大学教授、琶洲实验室（黄埔）主任）

大赛专家委员会

徐宗本（中国科学院院士、西安交通大学教授、琶洲实验室（黄埔）主任）

袁亚湘（中国科学院院士、中科院数学院研究员）

戴琼海（中国工程院院士、中国人工智能学会理事长、清华大学教授）

梅　宏（中国科学院院士、发展中国家科学院院士、北京大学教授）

王怀民（中国科学院院士、国防科技大学教授）

戴　浩（中国工程院院士、军事科学院系统工程研究院研究员）

周志华（欧洲科学院院士、国际人工智能联合会理事会主席、南京大学计算机系主任、人工智能学院院长）

石光明（鹏城实验室副主任、西安电子科技大学教授）

申恒涛（欧洲科学院院士、电子科技大学计算机科学与工程学院院长）

焦李成（欧洲科学院院士、西安电子科技大学计算机学部主任、人工智能研究院院长）

吴　枫（中国科学技术大学副校长）

姚　新（香港岭南大学副校长、南方科技大学计算机系主任）

刘铁岩（微软亚洲研究院副院长）

林宙辰（北京大学智能学院副院长）

郝志峰（汕头大学党委副书记、校长）

李树涛（湖南大学副校长）

杨　彤（欧洲科学院院士、发展中国家科学院院士、香港科学院院士、香港理工大学讲席教授）

张　潼（香港科技大学讲座教授）


大赛评测委员会

胡事民（中国科学院院士、中国计算机学会副理事长、清华大学教授）

王巨宏（腾讯公司副总裁）

陶大程（澳大利亚科学院院士、京东探索研究院院长）

田　奇（华为云人工智能领域首席科学家、国际欧亚科学院院士）

陈宝权（北京大学智能学院副院长）

戴礼荣（中国科学技术大学电子工程与信息科学系教授）

刘　挺（哈尔滨工业大学副校长）

程学旗（中国科学院计算所副所长）

陈　雷（香港科技大学（广州）信息枢纽院长）

李飞飞（阿里巴巴集团副总裁）

卢　凯（国防科技大学计算机学院院长）

金　海（中国计算机学会副理事长、华中科技大学教授）

王井东（百度计算机视觉首席科学家）

陈海波（上海交通大学并行与分布式系统研究所所长）

马建峰（西安电子科技大学网络空间安全学部主任）

田永鸿（北京大学计算机学院博雅特聘教授、北京大学深圳研究生院信息工程学院院长）

程明明（南开大学计算机系主任）

孟德宇（西安交通大学数学与统计学院教授、统计系系主任）

郑伟诗（中山大学计算机学院副院长）

左旺孟（哈尔滨工业大学计算机学院教授）

彭　玺（四川大学计算机学院教授）

宋井宽（电子科技大学计算机科学与工程学院教授）


算法大赛中心

申恒涛（欧洲科学院院士、算法大赛中心主任、电子科技大学计算机科学与工程学院院长、四川省人工智能研究院院长）

张　海（琶洲实验室（黄埔）主任助理、算法大赛中心常务副主任、西北大学数学学院教授）

徐　行（电子科技大学计算机科学与工程学院研究员）

谢晓华（中山大学计算机学院教授）


