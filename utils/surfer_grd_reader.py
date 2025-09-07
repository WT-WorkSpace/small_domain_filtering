import numpy as np

def _read_surfer6_text_grd(f):
    """
    读取GoldenSoftware Surfer的文本格式.grd文件
    参数:
        f(file object): 已经打开的文件对象
    返回:
        dict: 包含网格数据和元信息的字典，包括:
            - 'ncols': 列数
            - 'nrows': 行数
            - 'xmin': x坐标最小值
            - 'xmax': x坐标最大值
            - 'ymin': y坐标最小值
            - 'ymax': y坐标最大值
            - 'vmin': 网格数据最小值
            - 'vmax': 网格数据最大值
            - 'data': 包含网格数据的2D numpy数组
    """

    # 第1行：读取网格的列数和行数
    s = f.readline().strip().split() 
    ncols = int(s[0]) #列数在前
    nrows = int(s[1]) #行数在后
    # 第2行：xmin、xmax
    s = f.readline().strip().split() 
    xmin = float(s[0])
    xmax = float(s[1])
    # 第3行：ymin、ymax
    s = f.readline().strip().split() 
    ymin = float(s[0])
    ymax = float(s[1])
    # 第4行：vmin、vmax
    s = f.readline().strip().split() 
    vmin = float(s[0])
    vmax = float(s[1])
            
    # 读取数据
    values = [float(v) for v in f.read().split()]
            
    # 转换为numpy数组(2D)
    data = np.array(values,dtype=np.float32).reshape(nrows, ncols)
            
    # 构建返回结果
    result = {
        'ncols': ncols,
        'nrows': nrows,
        'xmin': xmin,
        'xmax': xmax,
        'ymin': ymin,
        'ymax': ymax,
        'vmin': vmin,
        'vmax': vmax,
        'data': data
    }
            
    return result

def _read_surfer6_binary_grd(f):
    """
    读取GoldenSoftware Surfer的二进制DSBB格式.grd文件
    参数:
        f(file object): 已经打开的文件对象
    返回:
        dict: 包含网格数据和元信息的字典，包括:
            - 'ncols': 列数
            - 'nrows': 行数
            - 'xmin': x坐标最小值
            - 'xmax': x坐标最大值
            - 'ymin': y坐标最小值
            - 'ymax': y坐标最大值
            - 'vmin': 网格数据最小值
            - 'vmax': 网格数据最大值
            - 'data': 包含网格数据的2D numpy数组
    """
    import sys
    native_order = sys.byteorder  # 多数系统(如Windows、Linux x86）值为 'little'
    
    # 读取网格的列数和行数（每个参数是2个字节的整数）
    ncols = int.from_bytes(f.read(2), byteorder=native_order) #number of grid lines along the X axis (columns)
    nrows = int.from_bytes(f.read(2), byteorder=native_order) #number of grid lines along the Y axis (rows)
    # 连续读取6个double：共48个字节（每个double是8个字节）
    min_max_array = np.frombuffer(f.read(6*8), dtype='f8')
    ''' 6个double对应信息如下：
    minimum X value of the grid
    maximum X value of the grid
    minimum Y value of the grid
    maximum Y value of the grid
    minimum Z value of the grid (注：z value 就是网格中的物理属性的数据，并不是z方向的坐标值)
    maximum Z value of the grid
    '''            
            
    # 读取数据部分（每个数据为4字节浮点数（单精度））
    data = np.frombuffer(f.read(4 * ncols * nrows), dtype='f4')
            
    # 转换为2D数组（数组中的第一行对应网格数据的最下面的一行）
    data = data.reshape(nrows, ncols)
            
    # 构建返回结果
    result = {
        'ncols': ncols,
        'nrows': nrows,
        'xmin': min_max_array[0],
        'xmax': min_max_array[1],
        'ymin': min_max_array[2],
        'ymax': min_max_array[3],
        'vmin': min_max_array[4],
        'vmax': min_max_array[5],
        'data': data
    }            
    return result

def read_surfer_grd(file_path):
    """
    读取GoldenSoftware Surfer6的.grd文件(只支持surfer6的text和binary这2种格式，还不支持surfer7 binary格式)
    参数:
        file_path (str): .grd文件的路径
    返回:
        dict: 包含网格数据和元信息的字典，包括:
            - 'ncols': 列数
            - 'nrows': 行数
            - 'xmin': x坐标最小值
            - 'xmax': x坐标最大值
            - 'ymin': y坐标最小值
            - 'ymax': y坐标最大值
            - 'vmin': 网格数据最小值
            - 'vmax': 网格数据最大值
            - 'data': 包含网格数据的2D numpy数组
    """
    try:
        with open(file_path, 'rb') as f:#注：不管是什么格式的grd文件，先总是用rb方式打开，用二进制方式读取前面4个字节的文件标识
            # 读取文件标识（最前面的4个字符） 
            header = f.read(4)
            if header == b'DSAA':
                f.close() #先关闭，重新以文本读取方式打开
                f = open(file_path, 'r')
                f.readline() #跳过第一行
                return _read_surfer6_text_grd(f)
            elif header == b'DSBB':
                return _read_surfer6_binary_grd(f)
            elif header == b'DSRB':
                print("该文件为surfer7 binary格式，暂不支持读取!")
                return None
            else:
                print("不支持的文件格式!")
                return None
            
    except Exception as e:
        print(f"读取文件时出错: {str(e)}")
        return None

if __name__ == "__main__":
    # 测试3种不同的格式
    surfer6_text_grd_file = "test_dsaa.grd"
    surfer6_bin_grd_file = "test_dsbb.grd"
    surfer7_bin_grd_file = "test_dsrb.grd"
    
    files = [surfer6_text_grd_file, surfer6_bin_grd_file, surfer7_bin_grd_file];

    for filepath in files:
        grd_data = read_surfer_grd(filepath)
    
        if grd_data:
            print(f"成功读取GRD文件:{filepath}")
            print(f"网格大小: {grd_data['nrows']}行 x {grd_data['ncols']}列")
            print(f"x坐标范围: ({grd_data['xmin']}, {grd_data['xmax']})")
            print(f"y坐标范围: ({grd_data['ymin']}, {grd_data['ymax']})")
            print(f"网格数据值范围: ({grd_data['vmin']}, {grd_data['vmax']})")
            print(f"数据数组形状: {grd_data['data'].shape}")    
            # 数据统计信息
            print("数据统计:")
            print(f"最小值: {np.nanmin( grd_data['data'])}")
            print(f"最大值: {np.nanmax( grd_data['data'])}")
            print(f"平均值: {np.nanmean(grd_data['data'])}")
            print("\n")
    