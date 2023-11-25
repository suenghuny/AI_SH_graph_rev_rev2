from Components.Modeler_Component_test import *
from Components.Adapter_Component import *
from Components.Policy import *
from cfg import get_cfg
import numpy as np

def simulation(solution):
    temperature1 = solution[0]
    interval_constant_blue1 = solution[1]
    temperature2 = solution[2]
    interval_constant_blue2 = solution[3]
    air_alert_distance = solution[4]
    score = 0
    n = 100
    seed = 4
    np.random.seed(seed)
    random.seed(seed)
    for e in range(n):
        env = modeler(data,
                      visualize=visualize,
                      size=size,
                      detection_by_height=detection_by_height,
                      tick=tick,
                      simtime_per_framerate=simtime_per_frame,
                      ciws_threshold=ciws_threshold,
                      action_history_step=cfg.action_history_step,
                      air_alert_distance_blue=  air_alert_distance,
                      interval_constant_blue = [interval_constant_blue1, interval_constant_blue2]
                      )
        epi_reward, eval, win_tag= evaluation(env, temperature1=temperature1,temperature2 = temperature2)
        if win_tag != 'lose':
            score += 1/n
        else:
            score += 0

    print(score, solution)
    return score

def fitness_func(ga_instance, solution, solution_idx):
    score = simulation(solution)
    return score

def preprocessing(scenarios):
    scenario = scenarios
    if mode == 'txt':
        input_path = ["Data/Test/dataset{}/ship.txt".format(scenario),
                      "Data/Test/dataset{}/patrol_aircraft.txt".format(scenario),
                      "Data/Test/dataset{}/SAM.txt".format(scenario),
                      "Data/Test/dataset{}/SSM.txt".format(scenario),
                      "Data/Test/dataset{}/inception.txt".format(scenario)]
    else:
        input_path = "Data\input_data.xlsx"

    data = Adapter(input_path=input_path,
                   mode=mode,
                   polar_chart=episode_polar_chart,
                   polar_chart_visualize=polar_chart_visualize)
    return data




def evaluation(env,
               temperature1,
               temperature2
               ):
    temp = random.uniform(0, 50)
    agent_blue = Policy(env, rule='rule3', temperatures=[temperature1, temperature2])
    agent_yellow = Policy(env, rule='rule2', temperatures=[temp, temp])
    done = False
    episode_reward = 0

    eval = False
    enemy_action_for_transition = [0] * len(env.enemies_fixed_list)
    friendly_action_for_transition = [0] * len(env.friendlies_fixed_list)
    step_checker = 0


    while not done:
        if env.now % (decision_timestep) <= 0.00001:
            avail_action_blue, target_distance_blue, air_alert_blue = env.get_avail_actions_temp(side='blue')
            avail_action_yellow, target_distance_yellow, air_alert_yellow = env.get_avail_actions_temp(side='yellow')

            action_blue = agent_blue.get_action(avail_action_blue, target_distance_blue, air_alert_blue)
            action_yellow = agent_yellow.get_action(avail_action_yellow, target_distance_yellow, air_alert_yellow)
            reward, win_tag, done, leaker = env.step(action_blue, action_yellow, rl = False)
            episode_reward += reward
            status = None
            step_checker += 1
        else:
            pass_transition = True
            env.step(action_blue=friendly_action_for_transition,
                     action_yellow=enemy_action_for_transition, pass_transition=pass_transition, rl = False)

        if done == True:
            break
    return episode_reward, eval, win_tag


if __name__ == "__main__":
    cfg = get_cfg()
    vessl_on = cfg.vessl
    if vessl_on == True:
        import vessl

        vessl.init()
        output_dir = "/output/"
        import os

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    else:
        from torch.utils.tensorboard import SummaryWriter

        output_dir = "../output_susceptibility_heuristic/"
        writer = SummaryWriter('./logs2')
        import os

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    import time

    """
    환경 시스템 관련 변수들
    """

    mode = 'txt'
    vessl_on = False
    polar_chart_visualize = False
    polar_chart_scenario1 = [33, 29, 25, 33, 30, 30, 55, 27, 27, 35, 25, 30, 40]  # RCS의 polarchart 적용

    polar_chart = [polar_chart_scenario1]
    df_dict = {}
    episode_polar_chart = polar_chart[0]
    datasets = [i for i in range(1, 2)]
    for dataset in datasets:
        data = preprocessing(dataset)
        visualize = False  # 가시화 기능 사용 여부 / True : 가시화 적용, False : 가시화 미적용
        size = [600, 600]  # 화면 size / 600, 600 pixel
        tick = 500  # 가시화 기능 사용 시 빠르기
        simtime_per_frame = cfg.simtime_per_frame
        decision_timestep = cfg.decision_timestep
        detection_by_height = False  # 고도에 의한
        num_iteration = cfg.num_episode  # 시뮬레이션 반복횟수
        rule = 'rule2'          # rule1 : 랜덤 정책 / rule2 : 거리를 기반 합리성에 기반한 정책(softmax policy)
        temperature = [10, 20]  # rule = 'rule2'인 경우만 적용 / 의사결정의 flexibility / 첫번째 index : 공중 위험이 낮은 상태, 두번째 index : 공중 위험이 높은 상태
        ciws_threshold = 0.5
        lose_ratio = list()
        remains_ratio = list()
        df_dict = {}
        records = list()

        solution_space = [[i for i in range(0, 20)], [i  for i in range(0, 50)],
                          [i  for i in range(0, 20)], [i  for i in range(0, 50)], [i for i in range(0,200)]]
        num_genes = len(solution_space)

        initial_population = []
        sol_per_pop = 20
        for _ in range(sol_per_pop):
            new_solution = [np.random.choice(space) for space in solution_space]
            initial_population.append(new_solution)
        #print(initial_population)

        num_generations = 100  # 세대 수
        num_parents_mating = 6  # 각 세대에서 선택할 부모 수

        sol_per_pop = 8

        init_range_low = 0
        init_range_high = 20

        parent_selection_type = "sss"
        keep_parents = 1

        crossover_type = "single_point"

        mutation_type = "random"
        mutation_percent_genes = 10

        import pygad
        ga_instance = pygad.GA(num_generations=num_generations,
                               num_parents_mating=num_parents_mating,
                               fitness_func=fitness_func,
                               sol_per_pop=sol_per_pop,
                               num_genes=num_genes,
                                parent_selection_type = parent_selection_type,
                                keep_parents = keep_parents,
                               initial_population=initial_population,
                               gene_space = solution_space,
                               crossover_type = crossover_type,
                               mutation_type = mutation_type,
                               mutation_percent_genes = mutation_percent_genes )

        # 최적화 실행
        ga_instance.run()
        fitness = ga_instance.best_solution()[1]
        best_solutions = ga_instance.best_solution()[0]

        fitness_history = ga_instance.get_fitness_history()
        plt.plot(fitness_history)
        plt.title("최적화 결과")
        plt.xlabel("세대")
        plt.ylabel("적합도")
        plt.show()

        print("최적해:", best_solutions)
        print("최적해의 적합도:", fitness)



