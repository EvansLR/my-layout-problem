
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


#编码空间的位数
DNA_SIZE = 24
#种群数目
POP_SIZE = 200
#杂交概率
CROSSOVER_RATE = 0.8
#变异概率
MUTATION_RATE = 0.006
#迭代次数
N_GENERATIONS = 50
#x定义域
X_BOUND = [-3, 3]
#y定义域
Y_BOUND = [-3, 3]

#目标函数，本程序求最大值
def F(x, y):
    return 3*(1-x)**2*np.exp(-(x**2)-(y+1)**2)- 10*(x/5 - x**3 - y**5)*np.exp(-x**2-y**2)- 1/3**np.exp(-(x+1)**2 - y**2)

#画图函数
def plot_3d(ax):
#    在python中，list变量前面加星号、字典变量前面加两个星号，列表前面加星号作用是将列表解开成 len(list) 个独立的参数，传入函数
#生成-3 到 3的一百个等距点
    X = np.linspace(*X_BOUND, 100)
    Y = np.linspace(*Y_BOUND, 100)
# 生成网格点坐标矩阵
    X,Y = np.meshgrid(X, Y)
#根据点生成Z轴的函数值
    Z = F(X, Y)
# rstride:行之间的跨度  cstride:列之间的跨度 cmap是颜色映射表，改变cmap参数可以控制三维曲面的颜色组合
    ax.plot_surface(X,Y,Z,rstride=1,cstride=1,cmap=cm.coolwarm)
#设置Z轴的区间
    ax.set_zlim(-10,10)
#给三个坐标轴分别打上对应标签
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
#停顿3秒，每次迭代产生不同的图片
    plt.pause(3)
#展示图片
    plt.show()

#获取适应度函数
def get_fitness(pop): 
    #利用解码函数将基因解码成十进制数字，从而计算适应度
    x,y = translateDNA(pop)
    #利用函数计算出适应度
    pred = F(x, y)
    #减去最小的适应度是为了防止适应度出现负数，通过这一步fitness的范围为[0, np.max(pred)-np.min(pred)],最后在加上一个很小的数防止出现为0的适应度
    return (pred - np.min(pred)) + 1e-3

#解码函数
def translateDNA(pop): #pop表示种群矩阵，一行表示一个二进制编码表示的DNA，矩阵的行数为种群数目
    #遍历每行，每列从此第一列开始，然后每次加2，即为所有的奇数列
    x_pop = pop[:,1::2]
    #遍历每行，每列从此第零列开始，然后每次加2，即为所有的偶数列
    y_pop = pop[:,::2] #偶数列表示y
    #pop:(POP_SIZE,DNA_SIZE)*(DNA_SIZE,1) --> (POP_SIZE,1)
    #矩阵点乘，进行解码
    x = x_pop.dot(2**np.arange(DNA_SIZE)[::-1])/float(2**DNA_SIZE-1)*(X_BOUND[1]-X_BOUND[0])+X_BOUND[0]
    y = y_pop.dot(2**np.arange(DNA_SIZE)[::-1])/float(2**DNA_SIZE-1)*(Y_BOUND[1]-Y_BOUND[0])+Y_BOUND[0]
    return x,y

#选择与杂交
def crossover_and_mutation(pop, CROSSOVER_RATE = 0.8):
    #产生的新种群
    new_pop = []
    #遍历种群中的每一个个体，将该个体作为父亲
    for father in pop:		
        #孩子先得到父亲的全部基因
        child = father		
        #产生子代时不是必然发生交叉，而是以一定的概率发生交叉
        if np.random.rand() < CROSSOVER_RATE:	
            #再种群中选择另一个个体，并将该个体作为母亲，POP_SIZE用来产生随机数进行选择
            mother = pop[np.random.randint(POP_SIZE)]	
            #随机产生交叉的点
            cross_points = np.random.randint(low=0, high=DNA_SIZE*2)	
            #孩子得到位于交叉点后的母亲的基因
            child[cross_points:] = mother[cross_points:]	
        mutation(child)	#每个后代有一定的机率发生变异
        #将新产生的个体加入到新种群中
        new_pop.append(child)
    return new_pop
#变异函数
def mutation(child, MUTATION_RATE=0.003):
    if np.random.rand() < MUTATION_RATE: 				#以MUTATION_RATE的概率进行变异
        #随机产生一个整数，代表要变异基因的位置
        mutate_point = np.random.randint(0, DNA_SIZE*2)
        #将变异点的二进制为反转，进行位运算 异或运算 0^1=1 1^1=0
        child[mutate_point] = child[mutate_point]^1 	

#选择函数（利用内置的轮盘赌函数）
def select(pop, fitness):    #轮盘赌的方法
    #以p概率选择区间[0,POPSIZE)内的个体，replace代表有放回
    idx = np.random.choice(np.arange(POP_SIZE), size=POP_SIZE, replace=True,p=(fitness)/(fitness.sum()))
    #返回选择后的种群
    return pop[idx]

#输出相关信息
def print_info(pop):
    #得到现在种群所有个体适应度
    fitness = get_fitness(pop)
    #得到适应度的最大值
    max_fitness_index = np.argmax(fitness)
    #输出最大值的大小
    print("max_fitness:", fitness[max_fitness_index])
    #将种群个体进行解码
    x,y = translateDNA(pop)
    #输出最优的基因型
    print("最优的基因型：", pop[max_fitness_index])
    #输出x和y值的大小
    print("(x, y):", (x[max_fitness_index], y[max_fitness_index]))


if __name__ == "__main__":
    #创建图像
    fig = plt.figure()
    #创建3D坐标轴
    ax = Axes3D(fig)	
    #将画图模式改为交互模式，程序遇到plt.show不会暂停，而是继续执行
    plt.ion()
    #绘制3D图像
    plot_3d(ax)
    #随机得到POP_SIZE大小的种群，由于是x和y两个变量，所以最后是*2的大小
    pop = np.random.randint(2, size=(POP_SIZE, DNA_SIZE*2)) #matrix (POP_SIZE, DNA_SIZE)
    #迭代N代
    for _ in range(N_GENERATIONS):
        #进行解码
        x,y = translateDNA(pop)
        #防止循环sca变量产生的影响
        if 'sca' in locals(): 
            sca.remove()
        #画散点图
        sca = ax.scatter(x, y, F(x,y), c='black', marker='o');
        #绘图
        plt.show()
        #防止变化过快
        plt.pause(0.1)
        #种群进行杂交和变异
        pop = np.array(crossover_and_mutation(pop, CROSSOVER_RATE))
        #F_values = F(translateDNA(pop)[0], translateDNA(pop)[1])#x, y --> Z matrix
        #计算适应度
        fitness = get_fitness(pop)
        #自然选择
        pop = select(pop, fitness) #选择生成新的种群
    #打印出循环完毕后种群的信息
    print_info(pop)
    #如果在脚本中使用ion()命令开启了交互模式，没有使用ioff()关闭的话，则图像会一闪而过，并不会常留。要想防止这种情况，需要在plt.show()之前加上ioff()命令。
    plt.ioff()
    #绘制3D图像
    plot_3d(ax)

#this is my second change for the  project to test if the git works properly.