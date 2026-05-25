#!/usr/bin/env python3

import argparse
import os
import random
import logging

from typing import Dict
import numpy as np
import torch
import habitat
from habitat import Env, logger
from arguments import get_args
from habitat.config.default import get_config

from agents.objnav_agent307 import ObjectNav_Agent
import cv2
from collections import defaultdict
from tqdm import tqdm, trange
import imageio

import threading
from multiprocessing import Process, Queue

import sys
sys.path.append(".")
import time

from utils.shortest_path_follower import ShortestPathFollowerCompat
from utils.task import PreciseTurn
from habitat.sims.habitat_simulator.actions import (
    HabitatSimActions,
    HabitatSimV1ActionSpaceConfiguration,
)

from constants import color_palette, category_to_id 
# Gui
import open3d.visualization.gui as gui

from utils.vis_gui import ReconstructionWindow

from web_display import get_web_display, start_web_server


# def generate_point_cloud(window):
def main(args, send_queue, receive_queue):


    
    args.exp_name = "objectnav-"+ args.detector

    log_dir = "{}/logs/{}/".format(args.dump_location, args.exp_name)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logging.basicConfig(
        filename=log_dir + "eval.log",
        level=logging.INFO)

    config = get_config(config_paths=["configs/"+ args.task_config])

    logging.info(args)
    # logging.info(config)

    random.seed(config.SEED)
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    torch.set_grad_enabled(False)
 

    config.defrost()
    config.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = args.gpu_id
    config.freeze()

    # print(config)
    env = Env(config=config)

    follower = ShortestPathFollowerCompat(
        env._sim, 0.1, False
    )

    args.turn_angle = config.SIMULATOR.TURN_ANGLE
    agent = ObjectNav_Agent(args, follower)

    agg_metrics: Dict = defaultdict(float)

    num_episodes = len(env.episodes)
    if args.episode_count > -1:
        num_episodes = min(args.episode_count, len(env.episodes))
    print("num_episodes: ", num_episodes)

    fail_case = {}
    fail_case['collision'] = 0
    fail_case['success'] = 0
    fail_case['detection'] = 0
    fail_case['exploration'] = 0

    count_episodes = 0
    # for count_episodes in trange(num_episodes):
    start = time.time()

    start_web_server(host='0.0.0.0', port=5000)
    web_display = get_web_display()
        
    # 更新初始统计

    img_count = 460

    # cross_floor_episodes = {} 

    # 加载跨楼层episode字典
    cross_floor_episodes = load_cross_floor_episodes("cross_floor_episodes_c.json")
    
    # 添加统计变量
    cross_floor_stats = {
        'total': 0,
        'success': 0,
        'spl_sum': 0
    }
    same_floor_stats = {
        'total': 0,
        'success': 0,
        'spl_sum': 0
    }

    while count_episodes < num_episodes:
        obs = env.reset()

        print("Scene ID:", env.current_episode.scene_id)
        scene_id = env.current_episode.scene_id
        scene_id = scene_id.split('/')[-1].split('.')[0]
        print("Episode ID:", env.current_episode.episode_id)
        episode_id = env.current_episode.episode_id

        is_cross_floor = False
        if scene_id in cross_floor_episodes:
            if episode_id in cross_floor_episodes[scene_id]:
                is_cross_floor = True



        initial_agent_state = env.sim.get_agent_state()
        robot_start_position = initial_agent_state.position
        robot_start_y = robot_start_position[1]  # Y坐标


        print(obs.keys())

        agent.reset(env)
        cate_object = category_to_id[obs['objectgoal'][0]]
        print(cate_object)

        count_steps = 0
        start_ep = time.time()
        while not env.episode_over:

            # dd_s_time = time.time()


            
  
            if count_steps > 10000:
                action = 0
            else:
                agent_state = env.sim.get_agent_state()



                start = time.time()
                # action = agent.act(env, obs, agent_state, send_queue, receive_queue)
                action = agent.act(env,obs, agent_state, send_queue, receive_queue)
                elapsed = time.time() - start

                print(f"agent.act 执行时间: {elapsed:.6f} 秒")
                
                

            image_annotated = transform_rgb_bgr(agent.annotated_image)  # 224*224*3
            web_display.update_rgb(image_annotated)

            map_vis = agent.obstacle_map * 255  # 反转：障碍物变0，可通行变255
            map_vis = map_vis.astype(np.uint8)
            map_vis = cv2.cvtColor(map_vis, cv2.COLOR_GRAY2BGR)  # 转BGR彩色图

            explored_i_values, explored_j_values = np.where(agent.explored_map == 1)
            if len(explored_i_values) > 0:
                map_vis[explored_i_values, explored_j_values] = [255, 255, 255]  

            # 把障碍物位置涂成红色（BGR中红色是(0,0,255)）
            map_vis[agent.obstacle_map == 1] = [0, 0, 255]


            
            camera_matrix_T = agent.get_transform_matrix(agent_state)
            camera_position = camera_matrix_T[:3, 3]
            obs_i = camera_position[0]*100 / agent.args.map_resolution + int(agent.origins_grid[0])
            obs_j = camera_position[2]*100 / agent.args.map_resolution + int(agent.origins_grid[1])

            # 在画圆之前转换类型
            cv2.circle(map_vis, (int(obs_j), int(obs_i)), 5, (255, 0, 0), -1)

            for stairs_id, stairs_pcd in agent.stairs_pcds_dict.items():
                points = np.asarray(stairs_pcd.points)
                obs_i_values = np.floor((points[:, 0])*100 / agent.args.map_resolution).astype(int) + int(agent.origins_grid[0])
                obs_j_values = np.floor((points[:, 2])*100 / agent.args.map_resolution).astype(int) + int(agent.origins_grid[1])
                # 在设置像素值之前，添加边界检查
                valid_indices = (obs_i_values >= 0) & (obs_i_values < map_vis.shape[0]) & \
                                (obs_j_values >= 0) & (obs_j_values < map_vis.shape[1])

                # 只处理有效索引
                valid_i = obs_i_values[valid_indices]
                valid_j = obs_j_values[valid_indices]

                if len(valid_i) > 0:
                    map_vis[valid_i, valid_j] = (255, 255, 0)
                # map_vis[obs_i_values, obs_j_values] = (255, 255, 0)
            goal_ys, goal_xs = np.where(agent.goal_map == 1)
            print(goal_xs)
            if len(goal_xs) > 0:
                goal_xs = int(np.mean(goal_xs))
                goal_ys = int(np.mean(goal_ys))
            cv2.circle(map_vis, (int(goal_xs), int(goal_ys)), 5, (0, 255, 0), -1)
            # map_vis[agent.goal_map == 1] = [0, 255, 0]
            if len(agent.plan_path_np) > 0 and agent.plan_path_np is not None:
                pass
                # plan_path_x = (plan_path_np[:, 0] - int(self.origins_grid[0])) * self.args.map_resolution / 100.0
                # plan_path_y = plan_path_np[:, 0] * 0
                # plan_path_z = (plan_path_np[:, 1] - int(self.origins_grid[1])) * self.args.map_resolution / 100.0 
                # path_i_values = np.floor((agent.plan_path_np[:, 0])*100 / agent.args.map_resolution).astype(int) + int(agent.origins_grid[0])
                # path_j_values = np.floor((agent.plan_path_np[:, 2])*100 / agent.args.map_resolution).astype(int) + int(agent.origins_grid[1])
                # map_vis[path_i_values,path_j_values] = (0, 255, 0)
                # path_i_values = np.floor((agent.plan_path_np[:, 0])*100 / agent.args.map_resolution).astype(int) + int(agent.origins_grid[0])
                # path_j_values = np.floor((agent.plan_path_np[:, 2])*100 / agent.args.map_resolution).astype(int) + int(agent.origins_grid[1])
                # print(agent.plan_path_np)
                
                # path_i_values = agent.plan_path_np[:, 0] #ffm
                # path_j_values = agent.plan_path_np[:, 1]
                
                # 使用循环绘制点，每个点更大更清晰
                # for i, j in zip(path_i_values, path_j_values):
                #     if 0 <= i < map_vis.shape[0] and 0 <= j < map_vis.shape[1]:
                #         cv2.circle(map_vis, (j, i), 2, (0, 255, 0), -1)  # 注意opencv是(x,y)即(j,i)
            
            web_display.update_map(map_vis)

            if action == None:
                print("action = None")
                continue

            # while not web_display.keyboard_flag :
            #     key_input = web_display.keyboard_input
                

            #     if key_input == "w":
            #         action = 1
            #         print("action: FORWARD")
            #         web_display.keyboard_flag = True
            #     elif key_input == "a":
            #         action = 2
            #         print("action: LEFT")
            #         web_display.keyboard_flag = True
            #     elif key_input == "d":
            #         action = 3
            #         print("action: RIGHT")
            #         web_display.keyboard_flag = True
            #     elif key_input == "u":
            #         action = 4
            #         agent.eve_angle += 30
            #         print("action: UP")
            #         web_display.keyboard_flag = True
            #     elif key_input == "j":
            #         action = 5
            #         agent.eve_angle -= 30
            #         print("action: DOWN")
            #         web_display.keyboard_flag = True
            #     elif key_input == "r":
            #         image_rgb = cv2.cvtColor(obs["rgb"], cv2.COLOR_BGR2RGB) 
            #         img_name =  '/workspace/VLN-Game/img/' + str(img_count) + '.jpg'
            #         cv2.imwrite(img_name, image_rgb, [cv2.IMWRITE_JPEG_QUALITY, 95])
            #         img_count = img_count + 1
                        
            #         print("save img！！！")
                    
            #     elif key_input == "f":
            #         action = 0
            #         print("action: FINISH")
            #         web_display.keyboard_flag = True

                time.sleep(0.1)  # 10ms延时，释放CPU
            # print(f"key_input: {key_input}")
            key_input = ''
            web_display.keyboard_input = ''
            web_display.keyboard_flag = False


            # if count_episodes <= 6:
            #     action = 0
            obs = env.step(action)

            # web_display.update_rgb(obs["rgb"])
            # image_rgb = cv2.cvtColor(obs["rgb"], cv2.COLOR_BGR2RGB) 
            # img_name =  '/workspace/VLN-Game/image/' + str(img_count) + '.jpg'
            # cv2.imwrite(img_name, image_rgb, [cv2.IMWRITE_JPEG_QUALITY, 95])
            # img_count = img_count + 1


            count_steps += 1
            
            # dd_e_time = time.time()
            # print(' time:%.3fs\n'%(dd_e_time - dd_s_time)) 

        ate_object = category_to_id[obs['objectgoal'][0]]
        print(cate_object)

        record_scene_result(scene_id,env.get_metrics()["success"],env.get_metrics()["spl"])
        print_scene_statistics(scene_data)



        end = time.time()
        time_elapsed = time.gmtime(end - start)
        log = " ".join([
            "Time: {0:0=2d}d".format(time_elapsed.tm_mday - 1),
            "{},".format(time.strftime("%Hh %Mm %Ss", time_elapsed)),
            "num timesteps {},".format(count_steps),
            "FPS {},".format(int(count_steps / (end - start_ep)))
        ]) + '\n'

        log += "Failed Case: collision/exploration/detection/success/total:"
        log += " {:.0f}/{:.0f}/{:.0f}/{:.0f}({:.0f}),".format(
            np.sum(fail_case['collision']),
            np.sum(fail_case['exploration']),
            np.sum(fail_case['detection']),
            np.sum(fail_case['success']),
            count_episodes) + '\n'
        
        metrics = env.get_metrics()
        for m, v in metrics.items():
            if isinstance(v, dict):
                for sub_m, sub_v in v.items():
                    agg_metrics[m + "/" + str(sub_m)] += sub_v
            else:
                agg_metrics[m] += v

        log += "Metrics: "
        log += ", ".join(k + ": {:.3f}".format(v / count_episodes) for k, v in agg_metrics.items()) + " ---({:.0f}/{:.0f})".format(count_episodes, num_episodes)

        print(log)
        print(env.get_metrics())
        # input()
        logging.info(log)

        if args.save_video:
            imageio.mimsave(video_save_path, frames, fps=2)
            print(f"Video saved to {video_save_path}")
     
        

    avg_metrics = {k: v / count_episodes for k, v in agg_metrics.items()}

    for stat_key in avg_metrics.keys():
        logger.info("{}: {:.3f}".format(stat_key, avg_metrics[stat_key]))

    return


def visualization_thread(send_queue, receive_queue):
    app = gui.Application.instance
    app.initialize()
    mono = app.add_font(gui.FontDescription(gui.FontDescription.MONOSPACE))
    app_win = ReconstructionWindow(args, mono, send_queue, receive_queue)
    app.run()


if __name__ == "__main__":
    args = get_args()
    args.detector = 'yolo'
    send_queue = Queue()
    receive_queue = Queue()

    if args.visualize:
        # Create a thread for the Open3D visualization
        visualization = threading.Thread(target=visualization_thread, args=(send_queue, receive_queue,))
        visualization.start()

    # Run ROS code in the main thread
    main(args, send_queue, receive_queue)