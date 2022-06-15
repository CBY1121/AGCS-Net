import numpy as np
# from scipy import stats
from scipy.stats import levene, ttest_ind

agcsnet_ACC = []
csnet_ACC = []
unet_res_ACC = []
unet_ACC = []

agcsnet_SEN = []
csnet_SEN = []
unet_res_SEN = []
unet_SEN = []


# Python SciPy 统计显著性检验(Statistical Significance Tests)
# https://www.cjavapy.com/article/1140/
# T检验用于确定两个变量的均值之间是否存在显着差异。 并让我们知道它们是否属于同一分布。
# 这是一条两尾测试
# 函数ttest_ind()接受两个大小相同的样本，并生成t统计量和p值的元组。


# 当两总体方差相等时，即具有方差齐性，可以直接检验。
# res = ttest_ind(v1, v2)
# res = ttest_ind(v1, v2).pvalue

# 同样地，返回结果会返回t值和p值。
# 当不确定两总体方差是否相等时，应先利用levene检验，检验两总体是否具有方差齐性。
# res = levene(v1, v2)

# 如果返回结果的p值远大于0.05，那么我们认为两总体具有方差齐性。
# 如果两总体不具有方差齐性，需要加上参数equal_val并设定为False。如下。
# res = ttest_ind(v1, v2, equal_var=False)

# Alpha值:阿尔法值是有意义的水平。
# 例如:要拒绝零假设，数据必须接近极限。通常取值为0.01、0.05或0.1。

# P值:表明数据实际上有多接近极限。比较P值和alpha值以建立统计显着性。
# 如果p值<= alpha，我们将拒绝原假设，并说该数据具有统计意义。否则，我们接受原假设。

# P值	         碰巧的概率	           对无效假设	    统计意义
# P＞0.1	碰巧出现的可能性大于5%	不能否定无效假设	两组差别无显著意义
# P＜0.05	碰巧出现的可能性小于5%	可以否定无效假设	两组差别有显著意义
# P＜0.01	碰巧出现的可能性小于1%	可以否定无效假设	两者差别有非常显著意义


print("ROSE_ACC")
res12 = levene(agcsnet_ROSE_ACC, csnet_ROSE_ACC)
# print(res12)
res13 = ttest_ind(agcsnet_ROSE_ACC, csnet_ROSE_ACC)
print(res13)

res14 = levene(agcsnet_ROSE_ACC, unet_res_ROSE_ACC)
# print(res14)
res15 = ttest_ind(agcsnet_ROSE_ACC, unet_res_ROSE_ACC)
print(res15)

res16 = levene(agcsnet_ROSE_ACC, unet_ROSE_ACC)
# print(res16)
res17 = ttest_ind(agcsnet_ROSE_ACC, unet_ROSE_ACC)
print(res17)

# res32 = levene(mstcgnet_ROSE_ACC, utcnet_ROSE_ACC)
# print(res32)
# res33 = ttest_ind(mstcgnet_ROSE_ACC, utcnet_ROSE_ACC)
# print(res33)
#
# res34 = levene(mstcgnet_ROSE_ACC, utgnet_ROSE_ACC)
# print(res34)
# res35 = ttest_ind(mstcgnet_ROSE_ACC, utgnet_ROSE_ACC)
# print(res35)

print("ROSE_SEN")
res18 = levene(agcsnet_ROSE_SEN, csnet_ROSE_SEN)
# print(res18)
res19 = ttest_ind(agcsnet_ROSE_SEN, csnet_ROSE_SEN, equal_var=False)
print(res19)

res20 = levene(agcsnet_ROSE_SEN, unet_res_ROSE_SEN)
# print(res20)
res21 = ttest_ind(agcsnet_ROSE_SEN, unet_res_ROSE_SEN)
print(res21)

res22 = levene(agcsnet_ROSE_SEN, unet_ROSE_SEN)
# print(res22)
res23 = ttest_ind(agcsnet_ROSE_SEN, unet_ROSE_SEN, equal_var=False)
print(res23)

# res36 = levene(mstcgnet_ROSE_SEN, utcnet_ROSE_SEN)
# print(res36)
# res37 = ttest_ind(mstcgnet_ROSE_SEN, utcnet_ROSE_SEN)
# print(res37)
#
# res38 = levene(mstcgnet_ROSE_SEN, utgnet_ROSE_SEN)
# print(res38)
# res39 = ttest_ind(mstcgnet_ROSE_SEN, utgnet_ROSE_SEN)
# print(res39)

res40 = levene(agcsnet_ACC, csnet_ACC)
# print(res40)
res41 = ttest_ind(agcsnet_ACC, csnet_ACC, equal_var=False)
print(res41)

res42 = levene(agcsnet_ACC, unet_res_ACC)
# print(res42)
res43 = ttest_ind(agcsnet_ACC, unet_res_ACC)
print(res43)

res44 = levene(agcsnet_ACC, unet_ACC)
# print(res44)
res45 = ttest_ind(agcsnet_ACC, unet_ACC)
print(res45)

res46 = levene(agcsnet_SEN, csnet_SEN)
# print(res46)
res47 = ttest_ind(agcsnet_SEN, csnet_SEN, equal_var=False)
print(res47)

res48 = levene(agcsnet_SEN, unet_res_SEN)
# print(res48)
res49 = ttest_ind(agcsnet_SEN, unet_res_SEN, equal_var=False)
print(res49)

res50 = levene(agcsnet_SEN, unet_SEN)
# print(res50)
res51 = ttest_ind(agcsnet_SEN, unet_SEN, equal_var=False)
print(res51)