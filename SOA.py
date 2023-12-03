"""
File  : SOA.py
Author: ACHIEVE_DREAM
Date  : 2023/12/1
"""
import numpy as np
import matplotlib.pyplot as plt
from math import exp


def custom_fitness_func(x: np.ndarray) -> np.ndarray:
    """
    适应度函数
    :param x: 位置
    :return: 适应度
    """
    # axis=1表示按行求和, 即计算每个个体的适应度
    return (x ** 4).sum(axis=1)


# noinspection DuplicatedCode
class SOA:
    # 有没有食物的阈值
    food_threshold = 0.25
    # 温度适不适合交配的阈值
    temp_threshold = 0.6
    # 模式阈值，在下面会使用到，当产生的随机值小于模式阈值就进入战斗模式，否则就进入交配模式
    model_threshold = 0.6
    # 常量c1,下面计算食物的质量的时候会用到
    c1 = 0.5
    # 常量c2,下面更新位置的时候会用到
    c2 = 0.5
    # 常量c3,用于战斗和交配
    c3 = 2
    # 雄性占比
    male_ratio = 0.5
    # 一个非常接近0的数, 防止除0操作
    delta = np.spacing(1)

    def __init__(self, dim: int, max_iter: int, population_size: int, lower_bound, upper_bound, fitness_func):
        """
        初始化SOA算法的参数
        :param dim: 维度(即解决方案的维度, 多少个参数需要处理)
        :param max_iter: 最大迭代次数
        :param population_size: 种群大小
        :param lower_bound: 下界
        :param upper_bound: 上界
        :param fitness_func: 适应度函数
        """
        self.dim = dim
        self.max_iter = max_iter
        self.population_size = population_size
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.fitness_func = fitness_func
        # 初始化种群, shape: (population_size, ndim)
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        # 初始化个体适应度, shape: (population_size,)
        self.fitness: np.ndarray = self.fitness_func(self.population)
        # 计算出全局最佳适应度, 因为这个是第一次进行操作，不用和别的进行对比
        self.global_best_position = self.fitness.argmin()
        self.global_best_fitness = self.fitness[self.global_best_position]
        # 初始化当前最佳适应度和当前最佳位置
        (self.male_current_best_fitness, self.male_current_best_position, self.female_current_best_fitness,
         self.female_current_best_position) = 0, 0, 0, 0
        # 得到食物的位置，其实就是当前全局最佳适应度的位置, shape: (ndim, )
        self.food_position = self.population[self.global_best_position]
        # 分离种群: 雄性, 雌性
        self.male_nums = int(self.population_size * self.male_ratio)
        self.female_nums = self.population - self.male_nums
        self.male_population, self.female_population = np.split(self.population, 2)
        # 分离适应度: 雄性, 雌性
        self.male_fitness, self.female_fitness = np.split(self.fitness, 2)
        # 计算雄性和雌性种群中的个体最佳,以及其解决方案
        self.male_best_position = self.male_fitness.argmin()
        self.male_best_fitness = self.male_fitness[self.male_best_position]
        self.male_best_fitness_solution = self.male_population[self.male_best_position]
        self.female_best_position = self.female_fitness.argmin()
        self.female_best_fitness = self.female_fitness[self.female_best_position]
        self.female_best_fitness_solution = self.female_population[self.female_best_position]

    def snake_optimization(self) -> (np.ndarray, float, np.ndarray):
        """
        蛇行优化算法
        :return: (最佳解决方案, 最佳适应度, 每代最佳适应度)
        """
        # 用于记录更新后的位置
        male_population = np.zeros_like(self.male_population)
        female_population = np.zeros_like(self.female_population)
        # 记录每代最佳适应度
        best_fitness = np.zeros(self.max_iter)
        for t in range(self.max_iter):
            # 计算温度
            temp = exp(-t / self.max_iter)
            # 计算食物的质量
            food_quality = self.c1 * exp((t - self.max_iter) / self.max_iter)
            if food_quality > 1:
                food_quality = 1
            # 先判断食物的质量是不是超过了阈值, 没有, 就寻找食物
            if food_quality < self.food_threshold:
                # 雄性寻找
                self.__find_food(self.male_population, male_population)
                self.__find_food(self.female_population, female_population)
            else:
                # 当前有食物开始进入探索阶段
                # 先判断当前的温度是冷还是热
                if temp > self.temp_threshold:  # 表示当前是热的
                    # 热了就不进行交配，开始向食物的位置进行移动
                    self.__move_to_food(self.male_population, male_population, temp)
                    self.__move_to_food(self.female_population, female_population, temp)
                else:
                    # 如果当前的温度是比较的冷的，就比较适合战斗和交配
                    # 生成一个随机值来决定是要战斗还是要交配
                    if np.random.random() < self.model_threshold:
                        # 进入战斗模式
                        self.__fight(self.male_population, male_population, food_quality, True)
                        self.__fight(self.female_population, female_population, food_quality, False)
                    else:
                        # 当前将进入交配模式
                        self.__mate(self.male_population, male_population, self.female_population, food_quality, True)
                        self.__mate(self.female_population, female_population, self.male_population, food_quality,
                                    False)
                    # 产蛋
                    if np.random.random() <= 0.5:
                        # 产出雄性
                        self.__born(self.male_fitness, male_population)
                        # 产出雌性
                        self.__born(self.female_fitness, female_population)
            # 将更新后的位置进行处理
            self.__update_params(self.male_population, male_population, self.male_fitness)
            self.__update_params(self.female_population, female_population, self.female_fitness)
            # 得到雄性个体中的最佳适应度
            self.male_current_best_position = self.male_fitness.argmin()
            self.male_current_best_fitness = self.male_fitness[self.male_current_best_position]
            # 得到雄性个体中的最佳适应度
            self.female_current_best_position = self.female_fitness.argmin()
            self.female_current_best_fitness = self.female_fitness[self.female_current_best_position]
            # 判断是否需要更新雄性种群的全局最佳适应度
            if self.male_current_best_fitness < self.male_best_fitness:
                # 更新解决方案
                self.male_best_fitness_solution = self.male_population[self.male_current_best_position]
                # 更新最佳适应度
                self.male_best_fitness = self.male_current_best_fitness
            # 判断是否需要更新雌性种群的全局最佳适应度
            if self.female_current_best_fitness < self.female_best_fitness:
                # 更新解决方案
                self.female_best_fitness_solution = self.female_population[self.female_current_best_position]
                # 更新最佳适应度
                self.female_best_fitness = self.female_current_best_fitness
            # 判断是否需要更新全局最佳适应度
            best_fitness[t] = self.male_current_best_fitness \
                if self.male_current_best_fitness < self.female_current_best_fitness \
                else self.female_current_best_fitness
            # 更新全局最佳适应度（这里是非常的奇怪的，不进行判断就直接更新了，他就能确定本代的最佳一定是比上一代好！！！）
            if self.male_best_fitness < self.female_best_fitness:
                self.global_best_fitness = self.male_best_fitness
                self.food_position = self.male_best_fitness_solution
            else:
                self.global_best_fitness = self.female_best_fitness
                self.food_position = self.female_best_fitness_solution
        return self.food_position, self.global_best_fitness, best_fitness

    def __find_food(self, population: np.ndarray, new_population: np.ndarray):
        """
        寻找食物
        :param population: 种群
        :param new_population:  待更新的种群
        :return: None
        """
        population_size = population.shape[0]
        for i in range(population_size):
            for j in range(self.dim):
                # 取得一个随机个体
                random_index = np.random.randint(0, population_size)
                rand_individual = population[random_index]
                # 随机生成+或者是-,来判断当前的c2是取正还是负
                rand_sign = np.random.choice([-1, 1])
                # 计算Am,np.spacing(1)是为了防止进行除法运算的时候出现除0操作, Am = exp(-F_rand / (F_i + delta))
                Am = exp(-self.male_fitness[random_index] / (self.male_fitness[i] + self.delta))
                # 更新位置
                new_population[i, j] = rand_individual[j] + rand_sign * self.c2 * Am * np.random.uniform(
                    self.lower_bound, self.upper_bound)

    def __move_to_food(self, population: np.ndarray, new_population: np.ndarray, temp: float):
        """
        向食物的位置移动
        :param population:  种群
        :param new_population: 待更新的种群
        :param temp: 当前温度
        :return: None
        """
        population_size = population.shape[0]
        for i in range(population_size):
            for j in range(self.dim):
                # 随机生成+或者是-,来判断当前的c2是取正还是负
                rand_sign = np.random.choice([-1, 1])
                new_population[i, j] = self.food_position[j] + rand_sign * self.c3 * temp * np.random.random() * (
                        self.food_position[j] - population[i, j])

    def __fight(self, population: np.ndarray, new_population: np.ndarray, food_quality: float, is_male: bool):
        """
        战斗模式
        :param population: 种群
        :param new_population: 待更新的种群
        :param food_quality: 食物质量
        :param is_male: 是否是雄性
        :return: None
        """
        population_size = population.shape[0]
        if is_male:
            for i in range(population_size):
                for j in range(self.dim):
                    # 先计算当前雄性的战斗的能力
                    fight_m = exp(-self.female_best_fitness / (self.male_fitness[i] + self.delta))
                    new_population[i, j] = population[i, j] + self.c3 * fight_m * np.random.random() * (
                            food_quality * self.male_best_fitness_solution[j] - population[i, j])
        else:
            for i in range(population_size):
                for j in range(self.dim):
                    # 先计算当前雌性的战斗的能力
                    fight_f = exp(-self.male_best_fitness / (self.female_fitness[i] + self.delta))
                    new_population[i, j] = population[i, j] + self.c3 * fight_f * np.random.random() * (
                            food_quality * self.female_best_fitness_solution[j] - population[i, j])

    def __mate(self, population: np.ndarray, new_population: np.ndarray, another_population: np.ndarray,
               food_quality: float, is_male: bool):
        """
        交配模式
        :param population: 种群
        :param new_population: 待更新的种群
        :param another_population: 另一个种群(交配)
        :param food_quality: 食物质量
        :param is_male: 是否是雄性
        :return: None
        """
        population_size = population.shape[0]
        if is_male:
            for i in range(population_size):
                for j in range(self.dim):
                    # 计算交配能力
                    mate_m = exp(-self.female_fitness[i] / (self.male_fitness[i] + self.delta))
                    new_population[i, j] = population[i, j] + self.c3 * mate_m * np.random.random() * (
                            food_quality * another_population[i, j] - population[i, j])
        else:
            for i in range(population_size):
                for j in range(self.dim):
                    # 计算交配能力
                    mate_f = exp(-self.male_fitness[i] / (self.female_fitness[i] + self.delta))
                    new_population[i, j] = population[i, j] + self.c3 * mate_f * np.random.random() * (
                            food_quality * another_population[i, j] - population[i, j])

    def __born(self, fitness: np.ndarray, population: np.ndarray):
        """
        产蛋: 更新最差适应度个体的位置
        :param fitness: 种群适应度
        :param population: 种群
        :return: None
        """
        # 更新最差适应度个体的位置
        population[fitness.argmax()] = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)

    def __update_params(self, population: np.ndarray, new_population: np.ndarray, fitness: np.ndarray):
        """
        更新参数
        :param population: 种群
        :param new_population: 待更新的种群
        :param fitness: 当前种群适应度
        :return: None
        """
        flag_low = new_population < self.lower_bound
        flag_high = new_population > self.upper_bound
        # 如果在范围内就不用更新, 否则超出上界,那么更新为最大值,否则更新为最小值
        new_population[flag_low] = self.lower_bound
        new_population[flag_high] = self.upper_bound
        # 计算种群中每一个个体的适应度（这个是被更新过位置的）, shape: (population_size,)
        new_fitness = self.fitness_func(new_population)
        # 判断是否需要更改当前个体的历史最佳适应度
        flag = new_fitness < fitness
        fitness[flag] = new_fitness[flag]
        population[flag] = new_population[flag]


if __name__ == '__main__':
    soa = SOA(10, 1000, 30, -100, 100, custom_fitness_func)
    food, global_fitness, gene_best_fitness = soa.snake_optimization()
    plt.plot(gene_best_fitness)
    plt.title("SOA算法")
    plt.xlabel("迭代次数")
    plt.ylabel("适应度")
    print("最佳的解决方案：", food)
    print("最佳适应度：", global_fitness)
    plt.show()
