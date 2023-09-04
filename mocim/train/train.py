# -*- coding: utf-8 -*-

# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

#!/usr/bin/env python
import torch
import numpy as np
import json
import os
from datetime import datetime
import argparse
# from torch.utils.tensorboard import SummaryWriter

import sys
print(os.path.abspath("../.."))
sys.path.append(os.path.abspath("../.."))

from mocim import visualisation
from mocim.modules import trainer
from mocim.eval.metrics import start_timing, end_timing



def train(
    device,
    config_file,
    chkpt_load_file=None,
    incremental=True,
    # vis
    if_vis = False,
    show_obj=False,
    update_im_freq=50,
    update_mesh_freq=200,
    # save
    save_path=None,
    use_tensorboard=False,
):
    # init trainer-------------------------------------------------------------
    # 初始化一个训练器，加载了所有必要参数
    mocim_trainer = trainer.Trainer(
        device,
        config_file,
        chkpt_load_file=chkpt_load_file,
        incremental=incremental,
    )

    # saving init--------------------------------------------------------------
    # 建立保存目录
    save = save_path is not None
    save_final = True
    eval_final = True
    save_grid_pc = False
    if save_final:
        if save_path is not None:
            save_final_path = save_path
        else:
            now = datetime.now()
            time_str = now.strftime("%m-%d-%y_%H-%M-%S")
            save_final_path = "../../results/" + time_str
        slice_final_path = os.path.join(save_final_path, 'slices')
        os.makedirs(slice_final_path)
        mesh_final_path = os.path.join(save_final_path, 'meshes')
        os.makedirs(mesh_final_path)
    if save_grid_pc:
        grid_pc_file = "../../results/"
    if save:
        # 把config文件写入文件夹
        with open(save_path + "/config.json", "w") as outfile:
            json.dump(mocim_trainer.config, outfile, indent=4)
        # 如果需要保存模型，见了checkpoints子文件夹
        if mocim_trainer.save_checkpoints:
            checkpoint_path = os.path.join(save_path, "checkpoints")
            os.makedirs(checkpoint_path)
        # 如果需要保存切片，建立slices子文件夹
        if mocim_trainer.save_slices:
            slice_path = os.path.join(save_path, 'slices')
            os.makedirs(slice_path)
            mocim_trainer.write_slices(
                slice_path, prefix="0.000_", include_gt=True)
        # 如果需要保存mesh，建立mesh子文件夹
        if mocim_trainer.save_meshes:
            mesh_path = os.path.join(save_path, 'meshes')
            os.makedirs(mesh_path)
        # writer = None
        # if use_tensorboard:
        #     writer = SummaryWriter(save_path)

    # eval init--------------------------------------------------------------
    # 如果要进行评估的话，设置好评估内容res
    if mocim_trainer.do_eval:
        res = [{} for _ in range(mocim_trainer.nr)]
        if mocim_trainer.sdf_eval:
            for i in range(mocim_trainer.nr):
                res[i]['sdf_eval'] = {}
        if mocim_trainer.mesh_eval:
            for i in range(mocim_trainer.nr):
                res[i]['mesh_eval'] = {}
    last_eval = [0 for _ in range(mocim_trainer.nr)]
    use_frame = [[] for _ in range(mocim_trainer.nr)]

    # main  loop---------------------------------------------------------------
    # 主循环部分，最多训练20000步，一般到不了
    print("Starting training for max", mocim_trainer.n_steps, "steps...")
    # 通过看轨迹长度，知道图像多少，将图像切分为两部分，给两个机器人使用
    size_dataset = len(mocim_trainer.scene_dataset)
    # 数据拆分为两段
    start_end = np.linspace(0,size_dataset,mocim_trainer.nr+1).astype(int)
    start = np.delete(start_end, -1)
    end = np.delete(start_end, 0)
    print("robot start from frame ",start)
    print("robot end with frame ",end)
    break_at = [-1 for _ in range(mocim_trainer.nr)]
    over = [False for _ in range(mocim_trainer.nr)]
    finish_optim = [False for _ in range(mocim_trainer.nr)]

    # 机器人一起迭代，t是共用的
    for t in range(mocim_trainer.n_steps):
        # break at end -------------------------------------------------------
        # 如果到了最后阶段，在保存一次最终结果，定量评估文件也需要再写一次
        for rid in range(mocim_trainer.nr):
            if t == break_at[rid]: 
                if save_final:
                    # 把当前slices保存下来
                    mocim_trainer.write_slices(slice_final_path, prefix="robot_"+str(rid)+"_",
                        include_gt=True, include_diff=False,
                        include_chomp=False, draw_cams=False,rid=rid)
                    mocim_trainer.write_mesh(mesh_final_path + "/robot_"+str(rid)+"_mesh.ply",rid=rid)
                if eval_final:
                    if mocim_trainer.sdf_eval and mocim_trainer.gt_sdf_file is not None:
                        visible_res = mocim_trainer.eval_sdf(visible_region=True,rid=rid)
                        print("robot_"+str(rid)+" final eval results: ************************:")
                        print("Visible region SDF error: {:.4f}".format(visible_res["av_l1"]))
                        print("Visible region SDF error using surf dis: {:.4f}".format(visible_res["surf_l1"]))
                        print("Visible region Bins error: ", visible_res["binned_l1"])
                        print("Visible region Chomp error: ", visible_res["l1_chomp_costs"])
                    if mocim_trainer.mesh_eval:
                        acc, comp = mocim_trainer.eval_mesh(rid=rid)
                        print("Mesh accuracy and completion:", acc, comp)
                    if mocim_trainer.do_eval:
                        # 把当前评估结果保存下来
                        # kf_list = mocim_trainer.frames.frame_id[:-1].tolist()
                        # res[rid]['kf_indices'] = kf_list
                        with open(os.path.join(save_path, "robot_"+str(rid)+'_res.json'), 'w') as f:
                            json.dump(res[rid], f, indent=4)
                if save_grid_pc:
                    # 评估用到的
                    sdf = mocim_trainer.get_sdf_grid(rid=rid).cpu().numpy()
                    np.save(grid_pc_file+"/robot_"+str(rid)+"_grid_sdf.npy",sdf)
                over[rid]=True
                mocim_trainer.no_over[rid]=False

        # 全部机器人都结束，则停止运行
        if over == [True for _ in range(mocim_trainer.nr)]:
            print(size_dataset)
            print("All use frame is")
            for i in range(mocim_trainer.nr):
                print(use_frame[i])
            break

        for rid in range(mocim_trainer.nr):
            if over[rid]==True:
                continue
            # get/add data---------------------------------------------------------
            # 判断当前帧是否迭代结束
            finish_optim[rid] =  mocim_trainer.steps_since_frame[rid] == mocim_trainer.optim_frames[rid]
            # 如果是增量式，并且需要优化或第一帧
            if incremental and (finish_optim[rid] or t == 0):
                # 使用新帧执行n步后，检查是否将其添加到关键帧集合。
                if t == 0:
                    # 如果是第一帧，一定是关键帧
                    add_new_frame = True
                else:
                    # 如果上一帧是关键帧，需要更多的迭代，设置迭代次数再进行;否则加进来一个新的帧
                    add_new_frame = mocim_trainer.check_keyframe_latest(rid=rid)
                # 如果需要加入进来一个新的关键帧
                if add_new_frame:
                    # 如果已经达到关键帧指标，要加入最新的当前时刻的帧（太模拟实时了吧）
                    new_frame_id = start[rid]+mocim_trainer.get_latest_frame_id(rid=rid)
                    if new_frame_id >= end[rid]:
                        # 如果这个最新的已经超过数据个数了，就是结束了,此时再迭代400次停止
                        break_at[rid] = t + mocim_trainer.iters_last
                        print("**************************************",
                            "robot_"+str(rid)+"  End of sequence",
                            "**************************************")
                    else:
                        use_frame[rid].append(new_frame_id)
                        # 如果还没有结束，说明当前帧所处位置，和帧的id。两者其实是有关联的，差一个帧率作为倍数
                        print("robot_"+str(rid)+"  Total step time", mocim_trainer.tot_step_time[rid])
                        print("frame______________________", new_frame_id)
                        # 得到当前帧的数据，一个数据类型
                        frame_data = mocim_trainer.get_data([new_frame_id])
                        # 在训练器中添加当前的数据,但需要指明是哪个机器人
                        mocim_trainer.add_frame(frame_data, rid=rid)
                        if t == 0:
                            # 如果是第一帧，需要优化200次
                            mocim_trainer.last_is_keyframe[rid] = True
                            mocim_trainer.optim_frames[rid] = mocim_trainer.iters_first
            # optimisation step---------------------------------------------
            # 优化步骤，根据loss优化一次网络参数
            losses, step_time = mocim_trainer.step(t=t, rid=rid, multi=mocim_trainer.multi)
            # 返回状态
            status = [k + ': {:.6f}  '.format(losses[k]) for k in losses.keys()]
            status = "".join(status) + '-- Step time: {:.2f}  '.format(step_time)
            loss = losses['total_loss']
            print("robot_"+str(rid), t, status)

        # visualisation----------------------------------------------------------
        # 可视化部分
        # 如果需要更新图像
        if if_vis:
            # 两个t是一样的，用哪个都行
            if update_im_freq is not None and (t % update_im_freq == 0):
                # 如果t达到40倍数
                display = {}
                # 更新需要展示的图像，是所有关键帧，和最新的一帧
                mocim_trainer.update_vis_vars(rid=0)
                mocim_trainer.update_vis_vars(rid=1)
                # keyframes的数据，是个大数组，直接可以可视化，先不展示反投影
                # display["keyframes"] = mocim_trainer.frames_vis(rid=0)
                # slices的数据，是个大数组，可以直接可视化
                display["slices_0"] = mocim_trainer.slices_vis(rid=0)
                display["slices_1"] = mocim_trainer.slices_vis(rid=1)
                # 如果需要更新mesh，所以update_mesh_freq一定要是update_im_freq的倍数
                if update_mesh_freq is not None and (t % update_mesh_freq == 0):
                    # 绘制mesh
                    # 返回得到scene
                    scene_0 = mocim_trainer.draw_3D(rid=0, show_pc=False, show_grid_pc = False, show_mesh = t > 200, draw_cameras= t <= 200, camera_view=False, show_gt_mesh=False)
                    scene_1 = mocim_trainer.draw_3D(rid=1, show_pc=False, show_grid_pc = False, show_mesh = t > 200, draw_cameras= t <= 200, camera_view=False, show_gt_mesh=False)                  
                # 返回一次当前scene，用于可视化，但循环会继续
                display["scene_0"] = scene_0
                display["scene_1"] = scene_1
                yield display

        # evaluation -----------------------------------------------------
        # 如果达到评估时间了
        for rid in range(mocim_trainer.nr):
            elapsed_eval = mocim_trainer.tot_step_time[rid] - last_eval[rid]
            if mocim_trainer.do_eval and elapsed_eval > mocim_trainer.eval_freq_s:
                last_eval[rid] = mocim_trainer.tot_step_time[rid] - mocim_trainer.tot_step_time[rid] % mocim_trainer.eval_freq_s
                # 如果要进行sdf的评估
                if mocim_trainer.sdf_eval and mocim_trainer.gt_sdf_file is not None:
                    visible_res = mocim_trainer.eval_sdf(visible_region=True,rid=rid)
                    # obj_errors = mocim_trainer.eval_object_sdf()
                    print("robot_"+str(rid)+"    Time ---------------------------------------------------->", mocim_trainer.tot_step_time[rid])
                    print("Visible region SDF error: {:.4f}".format(visible_res["av_l1"]))
                    print("Visible region SDF error using surf dis: {:.4f}".format(visible_res["surf_l1"]))
                    print("Visible region Bins error: ", visible_res["binned_l1"])
                    print("Visible region Chomp error: ", visible_res["l1_chomp_costs"])
                    if save:
                        res[rid]['sdf_eval'][t] = {
                            'time': mocim_trainer.tot_step_time[rid],
                            'rays': visible_res,
                        }
                if mocim_trainer.mesh_eval:
                    acc, comp = mocim_trainer.eval_mesh(rid=rid)
                    print("Mesh accuracy and completion:", acc, comp)
                    if save:
                        res[rid]['mesh_eval'][t] = {
                            'time': mocim_trainer.tot_step_time,
                            'acc': acc,
                            'comp': comp,
                        }
                if save:
                    with open(os.path.join(save_path, "robot_"+str(rid)+'_res.json'), 'w') as f:
                        json.dump(res[rid], f, indent=4)


# 主函数，最开始运行的地方
if __name__ == "__main__":
    # 如果有gpu的话，使用cuda
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 表示第一堆随机数，桌
    seed = 1
    np.random.seed(seed)
    torch.manual_seed(seed)
    # python用于调取参数的解释器
    parser = argparse.ArgumentParser(description="iSDF.")
    # json配置文件
    parser.add_argument("--config", type=str, help="input json config")
    # 是否增量式，这肯定是呀，所以不加
    parser.add_argument(
        "-ni",
        "--no_incremental",
        action="store_false",
        help="disable incremental SLAM option",
    )
    # 是否是一个无头序列，是多个场景使用的，所以不加
    parser.add_argument(
        "-hd", "--headless",
        action="store_true",
        help="run headless (i.e. no visualisations)"
    )
    args = parser.parse_args()

    config_file = args.config
    headless = args.headless
    incremental = args.no_incremental
    # 加载chkpt，也就是网络
    chkpt_load_file = None

    # vis，可视化使用的，多久更新一次
    show_obj = False
    update_im_freq = 80 #40
    update_mesh_freq = 240 #200
    if headless:
        update_im_freq = None
        update_mesh_freq = None

    # save，保存用的
    save = True #False
    use_tensorboard = False # False
    if save:
        # 如果需要保存，会创建一个文件夹
        now = datetime.now()
        time_str = now.strftime("%m-%d-%y_%H-%M-%S")
        save_path = "../../results/" + time_str
        os.mkdir(save_path)
    else:
        save_path = None

    scenes = train(
        device,
        config_file,
        chkpt_load_file=chkpt_load_file,
        incremental=incremental,
        # vis
        if_vis = False,
        show_obj=show_obj,
        update_im_freq=update_im_freq,
        update_mesh_freq=update_mesh_freq,
        # save
        save_path=save_path,
        use_tensorboard=use_tensorboard,
    )

    
    if headless:
        # 如果是多个场景，不用可视化，继续下一个
        on = True
        while on:
            try:
                out = next(scenes)
            except StopIteration:
                on = False
    else:
        # 如果是一个场景
        n_cols = 2
        if show_obj:
            n_cols = 3
        # 可视化是一个2 x 2的opencv窗口，展示当前场景，但主循环还在继续
        tiling = (2, n_cols)
        visualisation.display.display_scenes(scenes, height=int(680 * 0.7), width=int(1200 * 0.7), tile=tiling
        )
