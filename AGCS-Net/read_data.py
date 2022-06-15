import os
import matplotlib.pyplot as plt
import glob
#
# is_show = True  # 是否图片展示，false为不演示只保存
# is_save = True # 是否保存为图片
# 存txt得文件夹，里边可以放多个txt文件,每个txt会生成一个图片,并保存
# data_path =  'C:\\Users\\jiangyingjie\\PycharmProjects\\unet-res\\acc'

# data_path1 = 'C:\\Users\\cby\\PycharmProjects\\Merge_UNet_CSNet_CSNetAG_Acc'
# data_path1 = 'D:\\OCT\\OCTA\\ttest\\OCTA_3M'
# data_path1 = 'D:\\OCT\\OCTA\\ttest\\OCTA_6M'
data_path1 = 'D:\\OCT\\OCTA\\AGCS-Net\\AGCS-Net_Ttest_Rose\\OCTA_下载的数据集'
file_list1 = glob.iglob('{}/data_test_average_acc_agcsnet.txt'.format(data_path1))
# file_list1 = glob.iglob('{}/data_test_average_sen_agcsnet.txt'.format(data_path1))
for file in file_list1:
    data_list = []
    fp = open(file)
    temp_list = {}
    epoch = 0
    for line in fp:
        line = line.replace('\n', '')
        line = line.strip()  # 去掉空格
        if len(line) == 0:
            continue
        data_list.append(float(line))
    fp.close()
    x1 = data_list[2:]
    # print(x1)
    flag = 0
    index1 = []
    loss_values1 = []
    for value in x1:
        if flag == 0:
            index1.append(int(value))
            flag = 1
        else:
            loss_values1.append(float(value))
            flag = 0
    print("agcsnet")
    print(loss_values1)

# data_path1 = 'C:\\Users\\cby\\PycharmProjects\\Generated_Curve_Acc'
file_list2 = glob.iglob('{}/data_test_average_acc_csnet.txt'.format(data_path1))
# file_list2 = glob.iglob('{}/data_test_average_sen_csnet.txt'.format(data_path1))
for file in file_list2:
    data_list = []
    fp = open(file)
    temp_list = {}
    epoch = 0
    for line in fp:
        line = line.replace('\n', '')
        line = line.strip()  # 去掉空格
        if len(line) == 0:
            continue
        data_list.append(float(line))
    fp.close()
    x2 = data_list[2:]
    # print(x2)
    flag = 0
    index2 = []
    loss_values2 = []
    for value in x2:
        if flag == 0:
            index2.append(int(value))
            flag = 1
        else:
            loss_values2.append(float(value))
            flag = 0
    print("csnet")
    print(loss_values2)

# data_path1 = 'C:\\Users\\cby\\PycharmProjects\\Generated_Curve_Acc'
file_list3 = glob.iglob('{}/data_test_average_acc_unet_res.txt'.format(data_path1))
# file_list3 = glob.iglob('{}/data_test_average_sen_unet_res.txt'.format(data_path1))
for file in file_list3:
    data_list = []
    fp = open(file)
    temp_list = {}
    epoch = 0
    for line in fp:
        line = line.replace('\n', '')
        line = line.strip()  # 去掉空格
        if len(line) == 0:
            continue
        data_list.append(float(line))
    fp.close()
    x3 = data_list[2:]
    # print(x3)
    flag = 0
    index3 = []
    loss_values3 = []
    for value in x3:
        if flag == 0:
            index3.append(int(value))
            flag = 1
        else:
            loss_values3.append(float(value))
            flag = 0
    print("unet_res")
    print(loss_values3)

# data_path1 = 'C:\\Users\\cby\\PycharmProjects\\Generated_Curve_Acc'
file_list4 = glob.iglob('{}/data_test_average_acc_unet.txt'.format(data_path1))
# file_list4 = glob.iglob('{}/data_test_average_sen_unet.txt'.format(data_path1))
for file in file_list4:
    data_list = []
    fp = open(file)
    temp_list = {}
    epoch = 0
    for line in fp:
        line = line.replace('\n', '')
        line = line.strip()  # 去掉空格
        if len(line) == 0:
            continue
        data_list.append(float(line))
    fp.close()
    x4 = data_list[2:]
    # print(x4)
    flag = 0
    index4 = []
    loss_values4 = []
    for value in x4:
        if flag == 0:
            index4.append(int(value))
            flag = 1
        else:
            loss_values4.append(float(value))
            flag = 0
    print("unet")
    print(loss_values4)

# # file_list5 = glob.iglob('{}/data_test_average_acc_utcnet.txt'.format(data_path1))
# file_list5 = glob.iglob('{}/data_test_average_sen_utcnet.txt'.format(data_path1))
# for file in file_list5:
#     data_list = []
#     fp = open(file)
#     temp_list = {}
#     epoch = 0
#     for line in fp:
#         line = line.replace('\n', '')
#         line = line.strip()  # 去掉空格
#         if len(line) == 0:
#             continue
#         data_list.append(float(line))
#     fp.close()
#     x5 = data_list[2:]
#     # print(x5)
#     flag = 0
#     index5 = []
#     loss_values5 = []
#     for value in x5:
#         if flag == 0:
#             index5.append(int(value))
#             flag = 1
#         else:
#             loss_values5.append(float(value))
#             flag = 0
#     print("utcnet")
#     print(loss_values5)
#
# # file_list6 = glob.iglob('{}/data_test_average_acc_utgnet.txt'.format(data_path1))
# file_list6 = glob.iglob('{}/data_test_average_sen_utgnet.txt'.format(data_path1))
# for file in file_list6:
#     data_list = []
#     fp = open(file)
#     temp_list = {}
#     epoch = 0
#     for line in fp:
#         line = line.replace('\n', '')
#         line = line.strip()  # 去掉空格
#         if len(line) == 0:
#             continue
#         data_list.append(float(line))
#     fp.close()
#     x6 = data_list[2:]
#     # print(x6)
#     flag = 0
#     index6 = []
#     loss_values6 = []
#     for value in x6:
#         if flag == 0:
#             index6.append(int(value))
#             flag = 1
#         else:
#             loss_values6.append(float(value))
#             flag = 0
#     print("utgnet")
#     print(loss_values6)

# plt.figure()
# plt.title("Test ACC on data set Ⅰ")
# plt.xlabel("epochs")
# plt.ylabel("ACC value")
# plt.plot(index1, loss_values1, label='U-Net', color='green', linewidth=1.0, linestyle='-')
# plt.plot(index2, loss_values2, label='U-Net_Res', color='blue', linewidth=1.0, linestyle='-')
# plt.plot(index3, loss_values3, label='CS-Net', color='gray', linewidth=1.0, linestyle='-')
# plt.plot(index4, loss_values4, label='AGCS-Net', color='red', linewidth=1.0, linestyle='-')
# plt.legend(loc='lower right')

# if is_save:
#     file_name = (file.split(".")[0]).split('\\')[-1]
#     plt.savefig('merge_acc.png'.format(file_name))
# if is_show:
#     plt.show()
# plt.close('all')