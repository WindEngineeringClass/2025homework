import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import norm

# 加载数据并添加异常处理
def load_data(file_path):
    try:
        simData = sio.loadmat(file_path, squeeze_me=True, struct_as_record=False)
        return simData
    except FileNotFoundError:
        print(f"文件 {file_path} 未找到，请检查路径！")
        return None
    except Exception as e:
        print(f"加载数据时发生错误: {e}")
        return None

# 绘制平均速度
def plot_mean_velocity(U, heights, Ur=50, zr=160, alpha=0.22):
    # 计算实测平均值
    U_avg_measured = np.mean(U, axis=1)

    # 计算理论模型
    z_values = np.array(heights)
    U_avg_theory = Ur * (z_values / zr)**alpha

    # 绘图
    plt.figure(figsize=(10, 6))
    plt.plot(U_avg_measured, heights, 'bo-', label='Measured')
    plt.plot(U_avg_theory, heights, 'r--', label='Theory')
    plt.xlabel('Mean Velocity (m/s)')
    plt.ylabel('Height (m)')
    plt.title('Mean Velocity Profile (z-direction)')
    plt.legend()
    plt.grid()
    plt.gca().invert_yaxis()
    plt.show()

# 计算湍流强度并绘制
def turbulence_intensity(U, V, W, heights, I10=0.23, alpha=0.22):
    # 计算脉动速度标准差
    u_prime = U - np.mean(U, axis=1, keepdims=True)
    v_prime = V - np.mean(V, axis=1, keepdims=True)
    w_prime = W - np.mean(W, axis=1, keepdims=True)

    Iu_measured = np.std(u_prime, axis=1) / np.mean(U, axis=1)
    Iv_measured = np.std(v_prime, axis=1) / np.mean(U, axis=1)
    Iw_measured = np.std(w_prime, axis=1) / np.mean(U, axis=1)

    # 理论湍流强度
    z = np.array(heights)
    Iu_theory = I10 * (z / 10)**(-alpha)
    Iv_theory = 0.78 * Iu_theory
    Iw_theory = 0.55 * Iu_theory

    # 绘图
    plt.figure(figsize=(12, 6))
    plt.subplot(131)
    plt.plot(Iu_measured, heights, 'bo-', label='Measured')
    plt.plot(Iu_theory, heights, 'r--', label='Theory')
    plt.title('Iu(z)')
    
    plt.subplot(132)
    plt.plot(Iv_measured, heights, 'go-', label='Measured')
    plt.plot(Iv_theory, heights, 'r--', label='Theory')
    plt.title('Iv(z)')
    
    plt.subplot(133)
    plt.plot(Iw_measured, heights, 'mo-', label='Measured')
    plt.plot(Iw_theory, heights, 'r--', label='Theory')
    plt.title('Iw(z)')
    
    plt.tight_layout()
    plt.show()

# 主要执行逻辑
def main():
    # 数据加载
    simDataPath = './windData/zDirData.mat'
    simData = load_data(simDataPath)
    
    if simData is not None:
        # 提取数据
        U = simData['U']
        V = simData['V']
        W = simData['W']
        Y = simData['Y']
        Z = simData['Z']
        dt = simData['dt']
        heights_z = [10, 30, 50, 70, 90]  # z方向测点高度
        
        # 绘制平均速度
        plot_mean_velocity(U, heights_z)
        
        # 计算并绘制湍流强度
        turbulence_intensity(U, V, W, heights_z)

if __name__ == "__main__":
    main()
