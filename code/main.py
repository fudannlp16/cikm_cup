# -*- coding: utf-8 -*-
# @Time    : 2018/8/3 11:04
# @Author  : Xiaoyu Liu
# @Email   : liuxiaoyu16@fudan.edu.com


use_transfer_learning=True
if use_transfer_learning:
    import train_ss_cv
    import predict_ss
    train_ss_cv.train()
    predict_ss.predict()
else:
    import train_cv
    import predict
    train_cv.train()
    predict.predict()
