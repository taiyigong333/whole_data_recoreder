import pyrealsense2 as rs
import numpy as np
import cv2
import os
import time

if __name__ == "__main__":
    # 1. 创建保存图片的文件夹
    save_path = "./captured_images"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 2. 查询当前连接的所有 RealSense 设备
    ctx = rs.context()
    devices = ctx.query_devices()
    if len(devices) == 0:
        print("❌ 未检测到任何 RealSense 设备！")
        exit()

    print(f"✅ 找到 {len(devices)} 个设备:")
    cameras = []
    for dev in devices:
        name = dev.get_info(rs.camera_info.name)
        serial = dev.get_info(rs.camera_info.serial_number)
        print(f"  - {name} (S/N: {serial})")
        cameras.append({'name': name, 'serial': serial})

    # 3. 为每个相机创建独立的 pipeline、config 和 align
    pipelines = []
    
    for cam in cameras:
        pipeline = rs.pipeline()
        config = rs.config()
        
        # 核心：绑定特定的相机序列号
        config.enable_device(cam['serial'])
        
        # 只用彩色流，640x480 足够采集（最终 resize 到 256x256）
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        
        try:
            pipeline.start(config)
            pipelines.append({
                'name': cam['name'],
                'serial': cam['serial'],
                'pipe': pipeline,
            })
            print(f"▶️ {cam['name']} 启动成功!")
        except Exception as e:
            print(f"❌ 启动 {cam['name']} (S/N: {cam['serial']}) 失败: {e}")

    if not pipelines:
        print("没有相机成功启动，程序退出。")
        exit()

    print("\n--- 程序已启动 ---")
    print("操作说明: 按 's' 拍照保存, 按 'q' 或 'ESC' 退出\n")

    try:
        while True:
            color_images = []

            valid_frames = 0

            # 4. 遍历读取每个相机的帧
            for cam_data in pipelines:
                pipeline = cam_data['pipe']

                try:
                    frames = pipeline.wait_for_frames(timeout_ms=5000)
                except RuntimeError:
                    continue

                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue

                valid_frames += 1

                color_image = np.asanyarray(color_frame.get_data())

                cam_name = cam_data['name']
                cv2.putText(color_image, f"{cam_name} RGB", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                color_images.append(color_image)

            # 只有当两个相机都取到帧时才刷新显示
            if valid_frames != len(pipelines):
                continue

            # 5. 图像拼接与显示
            display_rows = []
            for i in range(len(pipelines)):
                color_resized = cv2.resize(color_images[i], (640, 360))
                display_rows.append(color_resized)

            final_display = np.vstack(display_rows) 

            cv2.imshow('RealSense Dual Camera', final_display)
            key = cv2.waitKey(1)

            # --- 拍照保存逻辑 ---
            if key & 0xFF == ord('s'):
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                for i, cam_data in enumerate(pipelines):
                    safe_name = cam_data['name'].replace(" ", "_")
                    color_file = os.path.join(save_path, f"{safe_name}_color_{timestamp}.png")
                    cv2.imwrite(color_file, color_images[i])

                print(f"抓拍成功！已保存 {len(pipelines)} 台相机的图片至: {save_path}")

            # 退出逻辑
            elif key & 0xFF == ord('q') or key == 27:
                break

    finally:
        # 关闭所有管道
        for cam_data in pipelines:
            cam_data['pipe'].stop()
        cv2.destroyAllWindows()