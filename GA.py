# -*- coding:utf-8 -*-

import random
import math
from RNN_3 import sess
from operator import itemgetter
import numpy as np
import tensorflow as tf
import random as rand
from tensorflow.python import debug as tf_debug
import matplotlib.pyplot as plt


class Gene:
    '''''
    遗传算法中基因（个体）的类
    每个对象都有两个属性：数据、大小
    '''

    def __init__(self, **data):
        self.__dict__.update(data)
        self.size = len(data['data'])  # length of gene

def evaluate(geneinfo):
    sess.run(pred, feed_dict={input: X_test, seqlen: [1, 2, 3, 4, 5, 6, 7]})
    return 1

class GA:
    '''''
    GA算法的类
    '''

    def __init__(self, parameter):
        '''''
        Initialize the pop of GA algorithom and evaluate the pop by computing its' fitness value .
        The data structure of pop is composed of several individuals which has the form like that:

        {'Gene':a object of class Gene, 'fitness': 1.02(for example)}

        Representation of Gene is a list: [b s0 u0 sita0 s1 u1 sita1 s2 u2 sita2]

        '''
        # parameter = [CXPB, MUTPB, NGEN, popsize, low, up]
        self.parameter = parameter

        low = self.parameter[4]
        up = self.parameter[5]

        self.bound = []
        self.bound.append(low)
        self.bound.append(up)

        pop = []
        for i in range(self.parameter[3]):
            geneinfo = []
            for pos in range(len(low)):
                geneinfo.append(random.uniform(self.bound[0][pos], self.bound[1][pos]))  # 初始化种群

            fitness = evaluate(geneinfo)  # 计算每个个体的适应度
            pop.append({'Gene': Gene(data=geneinfo), 'fitness': fitness})  # 存储染色体及其适应度

        self.pop = pop
        self.bestindividual = self.selectBest(self.pop)  # 存储种群中适应度最好的个体

    def selectBest(self, pop):
        '''''
        从序列中选出最佳个体
        '''
        s_inds = sorted(pop, key=itemgetter("fitness"), reverse=False)
        return s_inds[0]

    def selection(self, individuals, k):
        '''''
        从序列中选择两个个体
        '''
        s_inds = sorted(individuals, key=itemgetter("fitness"),
                        reverse=True)  # 按适应度的倒数进行排序 sort the pop by the reference of 1/fitness
        sum_fits = sum(1 / ind['fitness'] for ind in individuals)  # 对适应度的倒数求和 sum up the 1/fitness of the whole pop

        chosen = []
        for i in range(k):
            u = random.random() * sum_fits  # 在[0, sum_fits]范围内生成一个随机数 randomly produce a num in the range of [0, sum_fits]
            sum_ = 0
            for ind in s_inds:
                sum_ += 1 / ind['fitness']  # 对适应度的倒数求和 sum up the 1/fitness
                if sum_ > u:
                    # when the sum of 1/fitness is bigger than u, choose the one,
                    # which means u is in the range of [sum(1,2,...,n-1),sum(1,2,...,n)]
                    # and is time to choose the one ,namely n-th individual in the pop
                    chosen.append(ind)
                    break

        return chosen

    def crossoperate(self, offspring):
        '''''
        交叉操作 cross operation
        '''
        dim = len(offspring[0]['Gene'].data)

        geninfo1 = offspring[0]['Gene'].data  # Gene's data of first offspring chosen from the selected pop
        geninfo2 = offspring[1]['Gene'].data  # Gene's data of second offspring chosen from the selected pop

        pos1 = random.randrange(1, dim)  # select a position in the range from 0 to dim-1,
        pos2 = random.randrange(1, dim)

        newoff = Gene(data=[])  # offspring produced by cross operation
        temp = []
        for i in range(dim):
            if (i >= min(pos1, pos2) and i <= max(pos1, pos2)):
                temp.append(geninfo2[i])
                # the gene data of offspring produced by cross operation is from the second offspring in the range [min(pos1,pos2),max(pos1,pos2)]
            else:
                temp.append(geninfo1[i])
                # the gene data of offspring produced by cross operation is from the frist offspring in the range [min(pos1,pos2),max(pos1,pos2)]
        newoff.data = temp

        return newoff

    def mutation(self, crossoff, bound):
        '''''
        mutation operation
        '''

        dim = len(crossoff.data)

        pos = random.randrange(1, dim)  # chose a position in crossoff to perform mutation.

        crossoff.data[pos] = random.uniform(bound[0][pos], bound[1][pos])
        return crossoff

    def GA_main(self):
        '''''
        main frame work of GA
        '''

        popsize = self.parameter[3]

        print("Start of evolution")

        # Begin the evolution
        for g in range(NGEN):

            print("-- Generation %i --" % g)

            # Apply selection based on their converted fitness
            selectpop = self.selection(self.pop, popsize)

            nextoff = []
            while len(nextoff) != popsize:
                # Apply crossover and mutation on the offspring

                # 选择两个个体 Select two individuals
                offspring = [random.choice(selectpop) for i in range(2)]

                if random.random() < CXPB:  # 以CXPB的概率交叉两个个体 cross two individuals with probability CXPB
                    crossoff = self.crossoperate(offspring)
                    fit_crossoff = evaluate(crossoff.data)  # Evaluate the individuals

                if random.random() < MUTPB:  #i 以MUTPB的概率个体变异 mutate an individual with probability MUTPB
                    muteoff = self.mutation(crossoff, self.bound)
                    fit_muteoff = evaluate(muteoff.data)  # 评估个体 Evaluate the individuals
                    nextoff.append({'Gene': muteoff, 'fitness': fit_muteoff})

            # 种群完全演化到下一代 The population is entirely replaced by the offspring
            self.pop = nextoff

            # 统计全部适应度并打印 Gather all the fitnesses in one list and print the stats
            fits = [ind['fitness'] for ind in self.pop]

            length = len(self.pop)
            mean = sum(fits) / length
            sum2 = sum(x * x for x in fits)
            std = abs(sum2 / length - mean ** 2) ** 0.5
            best_ind = self.selectBest(self.pop)

            if best_ind['fitness'] < self.bestindividual['fitness']:
                self.bestindividual = best_ind

            print("  Best individual found is %s, %s" % (self.bestindividual['Gene'].data, self.bestindividual['fitness']))
            print("  Min fitness of current pop: %s" % min(fits))
            print("  Max fitness of current pop: %s" % max(fits))
            print("  Avg fitness of current pop: %s" % mean)
            print("  Std of currrent pop: %s" % std)

        print("-- End of (successful) evolution --")


if __name__ == "__main__":
    CXPB, MUTPB, NGEN, popsize = 0.8, 0.3, 50, 100  # 控制参数 control parameters

    up = [64, 64, 64, 64, 64, 64, 64]  # 变量上限 upper range for variables
    low = [-64, -64, -64, -64, -64, -64, -64]  # 变量下限 lower range for variables
    parameter = [CXPB, MUTPB, NGEN, popsize, low, up]

    run = GA(parameter)
    run.GA_main()