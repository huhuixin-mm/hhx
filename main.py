import cv2
import imageio
import numpy as np
import math
import time

# 取source与mask的交集
def intersectWithMask(img:np.ndarray, mask:np.ndarray, threshold:int=127, eps:float=1e-6)->np.ndarray:
    """
    Computes the intersection of the image and the mask.

    [img]  : np.ndarray, the image to be transformed
    [mask] : np.ndarray, the mask to be used for the transformation

    Returns:
    [ret]  : np.ndarray, the intersection of the image and the mask, shape (h, w, 3), dtype=np.float32
    """
    if img.shape[0:2] != mask.shape:
        raise ValueError("Image and mask must have the same shape\nshape of image: (_, _, _), shape of mask: (_, _)")
    ret = np.zeros(img.shape, dtype=np.float32)
    h, w = mask.shape

    for i in range(h):
        for j in range(w):
            k = 0 if mask[i][j] < 127 else 1 # threshold: 127
            ret[i, j] = img[i, j] if k else np.array([eps, eps, eps])
    return ret

# 计算投影矩阵
def computeProjectionMatrix(src_points:np.ndarray, dst_points:np.ndarray)->np.ndarray:
    """
    Computes the projection matrix from source points to destination points.

    [src_points] : np.ndarray, the source points, shape (4, 2)
    [dst_points] : np.ndarray, the destination points, shape (4, 2)

    Returns:
    [P]  : np.ndarray, the projection matrix
    """
    if src_points.shape!= (4, 2) or dst_points.shape!= (4, 2):
        raise ValueError("src_points and dst_points must be of shape (4, 2)")   
    A_list = []
    b_list = []

    for i in range(4): 
      A_list.append( [src_points[i, 0], src_points[i, 1], 1, 0, 0, 0, -dst_points[i, 0]*src_points[i, 0], -dst_points[i, 0]*src_points[i, 1]] )
      A_list.append( [0, 0, 0, src_points[i, 0], src_points[i, 1], 1, -dst_points[i, 1]*src_points[i, 0], -dst_points[i, 1]*src_points[i, 1]] )
      b_list.append(dst_points[i, 0])
      b_list.append(dst_points[i, 1])

    A = np.array(A_list)
    b = np.array(b_list)
    P = np.linalg.solve(A, b)
    P = np.concatenate((P, np.array([1])))
    return P.reshape((3, 3))

# 计算bicubic插值距离权重
def weight(dist:float, a=-0.5)->float:
    abs_dist = abs(dist)
    if abs_dist <= 1:  
        return (a + 2)*abs_dist**3 - (a + 3)*abs_dist**2 + 1
    elif abs_dist <= 2:  
        return a*abs_dist**3 - 5*a*abs_dist**2 + 8*a*abs_dist - 4*a
    else:  
        return 0 

# 变换图像, 并使用method进行插值
def transformImage(src:np.ndarray, dst:np.ndarray, P:np.ndarray, method:str)->np.ndarray:
    """
    Transforms the image using the projection matrix P and the specified method.

    [src]  : np.ndarray, the source image, shape (h, w, 3)
    [dst]  : np.ndarray, the destination image, shape (h, w, 3)
    [P]    : np.ndarray, the projection matrix, shape (3, 3)
    [method]: str, the interpolation method, can be 'nearest', 'bilinear', or 'bicubic'

    Returns:
    [ret]  : np.ndarray, the transformed image
    """
    h, w, _ = dst.shape
    ret = dst.copy()

    for i in range(h):
        for j in range(w):
            x, y, z = P @ np.array([i, j, 1])
            if z != 0:
                x /= z
                y /= z
            if x >= 0 and x < src.shape[0] and y >= 0 and y < src.shape[1]:   
                # 最近邻插值 
                if method == 'nearest':
                    ret[i, j] = src[int(x), int(y)]
                # 双线性插值
                elif method == 'bilinear':
                    x1, y1 = math.floor(x), math.floor(y)
                    x2, y2 = x1 + 1 if x1 < src.shape[0]-1 else x1, y1 + 1 if y1 < src.shape[1]-1 else y1
                    dx, dy = x - x1, y - y1
                    ret[i, j] = (1-dx) * (1-dy) * src[x1, y1] + dx * (1-dy) * src[x2, y1] + (1-dx) * dy * src[x1, y2] + dx * dy * src[x2, y2]
                # 双三次插值
                elif method == 'bicubic':
                    x_int, y_int = math.floor(x), math.floor(y) 

                    # 获取周围16个点的坐标  
                    x_range = list(range(x_int - 1, x_int + 3))  
                    y_range = list(range(y_int - 1, y_int + 3))    
      
                    # 计算加权和  
                    total = np.zeros(3)  
                    for i in range(4):  
                        for j in range(4):
                            # 处理边界  
                            x_fixed = min(max(x_range[i], 1), src.shape[0] - 2)  
                            y_fixed = min(max(y_range[j], 1), src.shape[1] - 2)   
                            total += src[x_fixed, y_fixed] * weight(x_range[i] - x) * weight(y_range[j] - y)  

                    # 计算插值结果  
                    ret[i, j] = total   
                else:
                    raise ValueError("Invalid method, it must be 'nearest', 'bilinear', or 'bicubic'")              
    return ret

# 将painter与图像叠加
def mergeImage(painter:np.ndarray, img:np.ndarray, eps:float=1e-6)->np.ndarray:
    """
    Merges the painter and the image.

    [painter] : np.ndarray, the painter image, shape (h, w, 3)
    [img]     : np.ndarray, the image to be merged, shape (h, w, 3)

    Returns:
    [ret]     : np.ndarray, the merged image
    """
    if painter.shape != img.shape:
        raise ValueError("Painter and image must have the same shape, shape (h, w, 3)")
    h, w, _ = img.shape
    ret = img.copy()
    for i in range(h):
        for j in range(w):
            if not np.all(np.isclose(painter[i, j], np.array([eps, eps, eps]))):
                ret[i, j] = painter[i, j]
    return ret.astype(np.uint8)

# 处理一张图像, 返回np.ndarray类型的结果
def process(src:np.ndarray, dst:np.ndarray, mask:np.ndarray, eps=1e-6, method='nearest'):
    # 取source与mask的交集
    painter = intersectWithMask(dst, mask, eps=eps)

    # 计算投影矩阵
    dst_points = np.array([[0, 0], [0, 660], [dst.shape[0], 0], [dst.shape[0], 688]])
    src_points = np.array([[0, 0], [0, src.shape[1]], [src.shape[0], 0], [src.shape[0], src.shape[1]]])
    P = computeProjectionMatrix(dst_points, src_points)

    transformed = transformImage(src, dst, P, method=method)

    # 将painter与图像叠加, 并输出结果
    merged = mergeImage(painter, transformed, eps=eps)
    return merged

if __name__ == '__main__':
    # 读取图像
    src_path = './images/source.png'
    mask = cv2.imread('./images/mask.png', 0) # 转换为灰度图
    dst = cv2.imread('./images/destination.png') # 读取目标图像
    eps = 1e-6 # 设定差量值, 用于判断是否为ROI
    method = 'bilinear' # 选择插值方法, 可选值: 'nearest', 'bilinear', 'bicubic'
    # 计时
    start_time = time.time()

    # 处理图片, 保存为.png格式
    if src_path.endswith('.png') or src_path.endswith('.jpg') or src_path.endswith('.jpeg'):
        src = cv2.imread(src_path)
        save_path = f'./result/{time.strftime("%Y%m%d_%H%M", time.localtime())}_by_{method}.png'
        ret = process(src, dst, mask, eps=eps, method=method)
        cv2.imwrite(save_path, ret)
    # 处理.mp4格式视频, 保存为.gif格式
    elif src_path.endswith('.mp4'):
        cap = cv2.VideoCapture(src_path)
        save_path = f'./result/{time.strftime("%Y%m%d_%H%M", time.localtime())}_by_{method}.gif'
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                out = process(src=frame, dst=dst, mask=mask, eps=eps, method=method)
                frames.append(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
            else:
                break
        cap.release()
        cv2.destroyAllWindows()
        frames += frames[-1::-1] # 倒放让视频看起来循环播放
        imageio.mimsave(save_path, frames, 'GIF', duration=0.02)
    else:
        raise ValueError("Invalid file type, it supports .png, .jpg, .jpeg, and .mp4 format up to now.")
        
    end_time = time.time()
    print("Time used: {:.2f}s".format(end_time - start_time))    