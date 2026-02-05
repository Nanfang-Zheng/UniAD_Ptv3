import os
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

def load_nuscenes_info(pkl_path: str) -> Dict[str, Any]:
    """
    加载nuScenes数据集的info文件
    
    Args:
        pkl_path: pkl文件路径
    
    Returns:
        包含数据集信息的字典
    """
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    return data

def check_file_exists(file_path: str) -> bool:
    """
    检查文件是否存在
    
    Args:
        file_path: 文件路径
    
    Returns:
        文件是否存在
    """
    return os.path.exists(file_path)

def extract_file_paths_from_nuscenes_info(info_data: Dict[str, Any], data_root: str) -> List[str]:
    """
    从nuScenes info数据中提取所有相关的文件路径
    
    Args:
        info_data: nuScenes info数据
        data_root: 数据根目录
    
    Returns:
        所有需要的文件路径列表
    """
    file_paths = []
    
    # 从infos中提取点云文件路径
    if 'infos' in info_data:
        for info in info_data['infos']:
            # 提取点云文件路径
            lidar_path = info.get('lidar_path', '')
            if lidar_path:
                # 如果路径是相对路径，则添加根目录
                if not lidar_path.startswith('/'):
                    full_path = os.path.join(data_root, lidar_path)
                else:
                    full_path = lidar_path
                file_paths.append(full_path)
            
            # 提取 sweeps 中的点云文件路径
            if 'sweeps' in info:
                for sweep in info['sweeps']:
                    sweep_path = sweep.get('data_path', '')
                    if sweep_path:
                        if not sweep_path.startswith('/'):
                            full_path = os.path.join(data_root, sweep_path)
                        else:
                            full_path = sweep_path
                        file_paths.append(full_path)
            
            # 提取相机图片路径
            if 'cams' in info:
                for cam_name, cam_data in info['cams'].items():
                    cam_path = cam_data.get('data_path', '')
                    if cam_path:
                        if not cam_path.startswith('/'):
                            full_path = os.path.join(data_root, cam_path)
                        else:
                            full_path = cam_path
                        file_paths.append(full_path)
    
    return file_paths

def check_nuscenes_dataset_completeness(pkl_paths: List[str], data_root: str) -> Dict[str, Any]:
    """
    检查nuScenes数据集完整性
    
    Args:
        pkl_paths: info pkl文件路径列表
        data_root: 数据根目录
    
    Returns:
        检查结果字典
    """
    all_file_paths = []
    
    # 从所有pkl文件中收集文件路径
    for pkl_path in pkl_paths:
        print(f"Loading info from {pkl_path}...")
        info_data = load_nuscenes_info(pkl_path)
        file_paths = extract_file_paths_from_nuscenes_info(info_data, data_root)
        all_file_paths.extend(file_paths)
    
    print(f"Total files to check: {len(all_file_paths)}")
    
    # 检查文件是否存在
    missing_files = []
    existing_files = []
    
    for i, file_path in enumerate(all_file_paths):
        if i % 1000 == 0:
            print(f"Checking file {i}/{len(all_file_paths)}...")
        
        if not check_file_exists(file_path):
            missing_files.append(file_path)
        else:
            existing_files.append(file_path)
    
    # 统计结果
    total_files = len(all_file_paths)
    missing_count = len(missing_files)
    existing_count = len(existing_files)
    
    result = {
        'total_files': total_files,
        'existing_files': existing_count,
        'missing_files': missing_count,
        'missing_file_list': missing_files,
        'existing_file_list': existing_files,
        'completeness_rate': existing_count / total_files if total_files > 0 else 0
    }
    
    return result

def analyze_missing_file_types(missing_files: List[str]) -> Dict[str, Any]:
    """
    分析缺失文件的类型分布
    
    Args:
        missing_files: 缺失文件列表
    
    Returns:
        文件类型分析结果
    """
    file_type_counts = {}
    lidar_top_count = 0
    camera_count = 0
    sweeps_count = 0
    samples_count = 0
    
    for file_path in missing_files:
        basename = os.path.basename(file_path)
        dirname = os.path.dirname(file_path)
        
        # 统计文件类型
        if 'LIDAR_TOP' in dirname:
            lidar_top_count += 1
        elif any(cam_type in dirname for cam_type in ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 
                                                      'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']):
            camera_count += 1
        
        if 'sweeps/' in dirname:
            sweeps_count += 1
        elif 'samples/' in dirname:
            samples_count += 1
            
        # 按扩展名统计
        ext = os.path.splitext(basename)[1].lower()
        file_type_counts[ext] = file_type_counts.get(ext, 0) + 1
    
    # 按目录路径统计
    dir_counts = {}
    for file_path in missing_files:
        dirname = os.path.dirname(file_path)
        # 只保留最后一级目录名
        last_dir = os.path.basename(dirname)
        dir_counts[last_dir] = dir_counts.get(last_dir, 0) + 1
    
    return {
        'lidar_top_count': lidar_top_count,
        'camera_count': camera_count,
        'sweeps_count': sweeps_count,
        'samples_count': samples_count,
        'file_type_counts': file_type_counts,
        'dir_counts': dir_counts
    }

def print_detailed_report(result: Dict[str, Any]):
    """
    打印详细的检查报告
    """
    print("\n" + "="*60)
    print("NU-SCENES DATASET COMPLETENESS REPORT")
    print("="*60)
    
    print(f"Total files to check: {result['total_files']}")
    print(f"Existing files: {result['existing_files']}")
    print(f"Missing files: {result['missing_files']}")
    print(f"Completeness rate: {result['completeness_rate']:.2%}")
    
    if result['missing_files'] > 0:
        # 分析缺失文件类型
        analysis = analyze_missing_file_types(result['missing_file_list'])
        
        print(f"\nFILE TYPE ANALYSIS:")
        print(f"  LIDAR_TOP files missing: {analysis['lidar_top_count']}")
        print(f"  Camera files missing: {analysis['camera_count']}")
        print(f"  Sweep files missing: {analysis['sweeps_count']}")
        print(f"  Sample files missing: {analysis['samples_count']}")
        
        print(f"\nFILE EXTENSION DISTRIBUTION:")
        for ext, count in analysis['file_type_counts'].items():
            print(f"  {ext}: {count} files")
        
        print(f"\nDIRECTORY DISTRIBUTION:")
        for dir_name, count in sorted(analysis['dir_counts'].items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {dir_name}: {count} files")
        
        print(f"\nALL MISSING FILES BY TYPE:")
        lidar_top_files = [f for f in result['missing_file_list'] if 'LIDAR_TOP' in f]
        camera_files = [f for f in result['missing_file_list'] if any(cam_type in f for cam_type in 
                    ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'])]
        other_files = [f for f in result['missing_file_list'] if f not in lidar_top_files and f not in camera_files]
        
        print(f"  LIDAR_TOP: {len(lidar_top_files)} files")
        print(f"  Camera: {len(camera_files)} files")
        print(f"  Other: {len(other_files)} files")
        
        print(f"\nAre ALL missing files LIDAR_TOP? {len(lidar_top_files) == len(result['missing_file_list'])}")
        
        print(f"\nFirst 10 missing files:")
        for i, missing_file in enumerate(result['missing_file_list'][:10]):
            file_type = "LIDAR_TOP" if "LIDAR_TOP" in missing_file else (
                "CAMERA" if any(cam_type in missing_file for cam_type in 
                               ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 
                                'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']) else "OTHER"
            )
            print(f"  {i+1}. [{file_type}] {missing_file}")
        
        if result['missing_files'] > 10:
            print(f"  ... and {result['missing_files'] - 10} more files")
    else:
        print("\n✅ All files are present! Dataset is complete.")
    
    print("="*60)

def check_specific_missing_file(missing_file_path: str, data_root: str) -> None:
    """
    检查特定缺失文件的详细信息
    """
    print(f"\nChecking specific missing file: {missing_file_path}")
    
    # 检查目录结构
    file_dir = os.path.dirname(missing_file_path)
    parent_dir = os.path.dirname(file_dir)
    
    print(f"File directory exists: {os.path.exists(file_dir)}")
    print(f"Parent directory exists: {os.path.exists(parent_dir)}")
    
    if os.path.exists(parent_dir):
        print(f"Contents of parent directory:")
        try:
            contents = os.listdir(parent_dir)
            for item in contents[:20]:  # 显示前20个文件
                print(f"  - {item}")
            if len(contents) > 20:
                print(f"  ... and {len(contents) - 20} more items")
        except Exception as e:
            print(f"Error listing directory: {e}")

# 示例使用
if __name__ == "__main__":
    # 根据配置文件，设置数据根目录
    data_root = '/home/zhengnanfang/PTv3_proj/UniAD'  # 或者 '/data/nuscenes-mini/' 如果你使用的是mini数据集
    
    # 设置info文件路径
    info_dir = '/home/zhengnanfang/PTv3_proj/UniAD/data/infos'
    pkl_files = [
        os.path.join(info_dir, 'nuscenes_infos_temporal_train.pkl'),
        os.path.join(info_dir, 'nuscenes_infos_temporal_val.pkl')
    ]
    
    # 检查数据集完整性
    result = check_nuscenes_dataset_completeness(pkl_files, data_root)
    
    # 打印报告
    print_detailed_report(result)
    
    # 如果有缺失文件，检查特定文件
    if result['missing_files'] > 0:
        # 检查你遇到的特定错误文件
        error_file = './data/nuscenes/sweeps/LIDAR_TOP/n008-2018-08-28-16-16-48-0400__LIDAR_TOP__1535488460646939.pcd.bin'
        abs_error_file = os.path.abspath(error_file)
        print(f"\nSpecific error file check: {abs_error_file}")
        print(f"File exists: {os.path.exists(abs_error_file)}")
        
        if not os.path.exists(abs_error_file):
            # 尝试检查相对路径
            alt_path = os.path.join(data_root, 'sweeps/LIDAR_TOP/n008-2018-08-28-16-16-48-0400__LIDAR_TOP__1535488460646939.pcd.bin')
            print(f"Alternative path check: {alt_path}")
            print(f"Alternative path exists: {os.path.exists(alt_path)}")