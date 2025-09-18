提交日期：2023年10月20日

团队名称与队长联系方式:Team_FeiYu, 郭良轩 中科院自动化所 guoliangxuan2021@ia.ac.cn

镜像说明：与上次评测时的镜像（第六周）无区别！！！使用评测专用镜像，python3.9.16，依赖包见Dockerfile和requirements.txt。cuda版本为11.7、torch2.0、numpy1.25.2时运行无错误

预计的训练时间:10splitTasks一个任务训练时间约为1小时30分钟，共约12小时。4splitDomains整个序列任务训练时间约为3小时。

其他细节说明: 训练时请在config.py中选择
                                    _C.AGENT.TYPE = 'trainer_OWM'
                                    _C.AGENT.NAME = 'OWMTrainer'

特殊说明： 
1. Q&A里说不允许修改CLTrainer.learn_task()方法，代码里有修改，但并未违反规则。连续学习策略为改进的OWM，使用训练完第一个任务的特征提取器，在后续任务上仅通过OWM和AOP正交修改的方式微调resnet50的最后三个卷积层和分类头。

######################################################
4splitDomains # 训练（复现）
Task 0
python iBatchLearn.py --cfg ./4splitDomains.yaml --user_cfg ./utils/user_4splitDomains.yaml --task_count 0 \
--suffix local_test --init_path ./init_model/4splitDomains.pth --ckpt_path ./init_model/4splitDomains.pth --save_ckpt_path ./model_info/4splitDomains/checkpoint-0.pth \
--save_storage_path ./model_info/4splitDomains/storage-0.pth

Task 1 其余任务同理
python iBatchLearn.py --cfg ./4splitDomains.yaml --user_cfg ./utils/user_4splitDomains.yaml --task_count 1 \
--suffix local_test --init_path ./init_model/4splitDomains.pth --ckpt_path ./model_info/4splitDomains/checkpoint-0.pth --save_ckpt_path ./model_info/4splitDomains/checkpoint-1.pth \
--storage_path ./model_info/4splitDomains/storage-0.pth --save_storage_path ./model_info/4splitDomains/storage-1.pth

4splitDomains # 推理（评审）
python iBatchLearn.py --cfg ./4splitDomains.yaml --user_cfg ./utils/user_4splitDomains.yaml --test --task_count 0 \
--suffix local_test --init_path ./init_model/4splitDomains.pth --ckpt_path ./model_info/4splitDomains/checkpoint-0.pth  \
--dest_path ./dest-4splitDomains.pkl

######################################################
10splitTasks # 训练（复现）
Task 0
python iBatchLearn.py --cfg ./10splitTasks.yaml --user_cfg ./utils/user_10splitTasks.yaml --task_count 0 \
--suffix local_test --init_path ./init_model/10splitTasks.pth --ckpt_path ./init_model/10splitTasks.pth --save_ckpt_path ./model_info/10splitTasks/checkpoint-0.pth \
--save_storage_path ./model_info/10splitTasks/storage-0.pth

Task 1 其余任务同理
python iBatchLearn.py --cfg ./10splitTasks.yaml --user_cfg ./utils/user_10splitTasks.yaml --task_count 1 \
--suffix local_test --init_path ./init_model/10splitTasks.pth --ckpt_path ./model_info/10splitTasks/checkpoint-0.pth --save_ckpt_path ./model_info/10splitTasks/checkpoint-1.pth \
--storage_path ./model_info/10splitTasks/storage-0.pth --save_storage_path ./model_info/10splitTasks/storage-1.pth

10splitTasks # 推理（评审）
python iBatchLearn.py --cfg ./10splitTasks.yaml --user_cfg ./utils/user_10splitTasks.yaml --test --task_count 0 \
--suffix local_test --init_path ./init_model/10splitTasks.pth --ckpt_path ./model_info/10splitTasks/checkpoint-0.pth  \
--dest_path ./dest-10splitTasks.pkl


