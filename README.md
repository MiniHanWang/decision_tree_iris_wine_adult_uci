#直接运行Run.py


# decision_tree_iris_wine_adult_uci
#@Time:10/26/20206:15 PM
#@Author: Mini(Wang Han)
#@Site:
#@File:run.py
from trees import *
#from cross_validation import cross_validation
import read_data
if __name__ == '__main__':
    decide_dataset = str(input("请选择数据集:->(1:iris; 2:wine; 3:adult)："))
    if decide_dataset == '1':
        print("选择数据集iris前5行：")
        dataset_, labels,labelProperties = read_data.read_iris()
        testFile="iris_test.xlsx"
        testSet_ = read_data.read_excel(testFile)
        testSet = read_data.random_select(testSet_, n=None, ratio=0.3)
        train=testSet.drop("class", axis=1, inplace=False)
        target=testSet["class"]
        test_name_ = "class"
        test_num=4
        classLabel="Iris-setosa"
    elif decide_dataset == "2":
        print("选择数据集wine前5行：")
        dataset_, labels, labelProperties = read_data.read_wine()
        testFile = "wine_test.xlsx"
        testSet_ = read_data.read_excel(testFile)
        testSet = read_data.random_select(testSet_, n=None, ratio=0.3)
        train =testSet.drop('Class', axis=1, inplace=False)
        target = testSet["Class"]
        test_name_ ="Class"
        test_num=0
        classLabel ="1"
    elif decide_dataset == "3":
        print("选择数据集adult前5行：")
        dataset_, labels,labelProperties = read_data.read_adult()
        testFile = "adult_test.xlsx"
        testSet_ = read_data.read_excel(testFile)
        testSet = read_data.random_select(testSet_, n=None, ratio=0.3)
        train = testSet.drop('weight', axis=1, inplace=False)
        target = testSet["weight"]
        test_name_="weight"
        test_num=14
        classLabel = "<=50K"

    dataset=dataset_.values.tolist()
    # dataset,features=createDataSet()
    print('dataset', dataset)
    print("---------------------------------------------")
    print(u"数据集长度", len(dataset))
    print("Ent(D):", Entropy_calculation(dataset))
    print("---------------------------------------------")

    print(u"以下为首次寻找最优索引:\n")
    print(u"ID3算法的最优特征索引为:" + str(ID3_chooseBestFeatureToSplit(dataset)))
    print("--------------------------------------------------")
    print(u"C4.5算法的最优特征索引为:" + str(C45_chooseBestFeatureToSplit(dataset)))
    print("--------------------------------------------------")
    print(u"CART算法的最优特征索引为:" + str(CART_chooseBestFeatureToSplit(dataset)))
    print(u"首次寻找最优索引结束！")
    print("---------------------------------------------")

    print(u"下面开始创建相应的决策树-------")

    while (True):
        dec_tree = str(input("请选择决策树:->(1:ID3; 2:C4.5; 3:CART)|('enter q to quit!')|："))
        # ID3决策树
        if dec_tree == '1':
            labels_tmp = labels[:]  # 拷贝
            ID3desicionTree = ID3_createTree(dataset, labels_tmp)

            #score=cross_validation(dataset_,ID3desicionTree,labels=labels_tmp,colunm_drop=test_num,colunm_drop_name=test_name_,k=10)
            print('ID3desicionTree:\n', ID3desicionTree)
            #print("cross_validation:",score)
            treePlotter.createPlot(ID3desicionTree)
            treePlotter.ID3_Tree(ID3desicionTree)

            print("下面为测试数据集结果：")
            print('ID3_TestSet_classifyResult:\n', classifytest(ID3desicionTree,train.columns.tolist(), train,classLabel ))
            print("---------------------------------------------")

        # C4.5决策树
        if dec_tree == '2':
            labels_tmp = labels[:]  # 拷贝
            C45desicionTree = C45_createTree(dataset, labels_tmp)
            #score = cross_validation(dataset_,C45desicionTree,labels=labels_tmp,colunm_drop=test_num,colunm_drop_name=test_name_, k=10)
            print('C45desicionTree:\n', C45desicionTree)
            #print("cross_validation:", score)
            treePlotter.C45_Tree(C45desicionTree)

            print("下面为测试数据集结果：")
            print('C4.5_TestSet_classifyResult:\n', classifytest(C45desicionTree, train.columns.tolist(), train,classLabel ))
            print("---------------------------------------------")

        # CART决策树
        if dec_tree == '3':
            labels_tmp = labels[:]  # 拷贝
            CARTdesicionTree = CART_createTree(dataset, labels_tmp)
            #score = cross_validation(dataset_,CARTdesicionTree,labels=labels_tmp,colunm_drop=test_num,colunm_drop_name=test_name_, k=10)
            print('CARTdesicionTree:\n', CARTdesicionTree)
            #print("cross_validation:", score)
            treePlotter.CART_Tree(CARTdesicionTree)

            print("下面为测试数据集结果：")
            print('CART_TestSet_classifyResult:\n', classifytest(CARTdesicionTree, train.columns.tolist(), train,classLabel ))
        if dec_tree == 'q':
            break

