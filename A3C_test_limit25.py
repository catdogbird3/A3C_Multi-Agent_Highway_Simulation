import numpy as np  # 引入NumPy庫
from hashlib import sha256  # 引入sha256散列函數
import hmac  # 引入HMAC庫
import random  # 引入隨機數生成庫
import hashlib  # 引入hashlib庫
import time  # 引入時間庫
import matplotlib.pyplot as plt  # 引入Matplotlib庫進行繪圖
import os  # 引入os庫生成隨機字節

# 定義參數
p = 8380417  # 定義大質數 p
m = 512  # 定義矩陣行數
n = 256  # 定義矩陣列數
kyber_k = 3  # Kyber768中的參數k
kyber_n = 256  # Kyber768中的參數n
kyber_eta = 2  # Kyber768中的參數eta

# 生成公開矩陣 M ∈ Z_p^{m×n} 並轉換為 NTT 格式
def generate_NTT_matrix(m, n, p):
    M_real = np.random.randint(0, p, size=(m, n))  # 生成實部隨機矩陣
    M_imag = np.random.randint(0, p, size=(m, n))  # 生成虛部隨機矩陣
    M = M_real + 1j * M_imag  # 合併實部和虛部
    M_NTT = np.fft.fft(M, axis=0)  # 將矩陣轉換為NTT格式
    return M_NTT  # 返回NTT格式矩陣

M_NTT = generate_NTT_matrix(m, n, p)  # 生成並轉換公開矩陣

# 生成密鑰對
def generate_keypair():
    random.seed(time.time_ns())  # 每次生成密鑰前重設隨機數種子
    alpha = random.getrandbits(256)  # 生成隨機256位元的alpha
    s = np.random.randint(-kyber_eta, kyber_eta + 1, size=(kyber_n, kyber_k))  # 生成s
    s_NTT = np.fft.fft(s, axis=0)  # 將s轉換為NTT格式
    e = np.random.randint(-kyber_eta, kyber_eta + 1, size=(m, kyber_k))  # 生成e
    e_NTT = np.fft.fft(e, axis=0)  # 將e轉換為NTT格式

    # 分開處理實數和虛數部分的餘數計算
    t_real = np.remainder(np.dot(M_NTT.real, s_NTT.real) - np.dot(M_NTT.imag, s_NTT.imag) + e_NTT.real, p)
    t_imag = np.remainder(np.dot(M_NTT.real, s_NTT.imag) + np.dot(M_NTT.imag, s_NTT.real) + e_NTT.imag, p)
    t = t_real + 1j * t_imag

    pk = (t, alpha)  # 公鑰包含t和alpha
    sk = s_NTT  # 私鑰為s_NTT
    return pk, sk  # 返回公鑰和私鑰

participants = ['Alice', 'Bob', 'Charlie', 'Dave']  # 定義參與者
keys = {participant: generate_keypair() for participant in participants}  # 為每個參與者生成密鑰對

# FACKA密鑰生成
def facka_key_generation(salt, IKM):
    PRK = hmac.new(salt, IKM, sha256).digest()  # 生成伺服器密鑰
    OKM = hmac.new(PRK, b"info" + b'\x01', sha256).digest()  # 生成輸出密鑰
    return PRK, OKM  # 返回伺服器密鑰和輸出密鑰

salt = b'random_salt'  # 定義隨機鹽值
IKM = b''.join([key[0][1].to_bytes(32, 'big') for key in keys.values()])  # 生成初始密鑰材料
PRK, OKM = facka_key_generation(salt, IKM)  # 生成FACKA密鑰

# 掩碼複數數值
def mask_complex(value, mask_value):
    real_masked = np.bitwise_xor(value.real.astype(np.int64), mask_value)  # 掩碼實部
    imag_masked = np.bitwise_xor(value.imag.astype(np.int64), mask_value)  # 掩碼虛部
    return real_masked + 1j * imag_masked  # 返回掩碼後的複數

# 檢查系統完整性
def check_system_integrity():
    code_content = """
import numpy as np  # 引入NumPy庫
from hashlib import sha256  # 引入sha256散列函數
import hmac  # 引入HMAC庫
import random  # 引入隨機數生成庫
import hashlib  # 引入hashlib庫

# 定義參數
p = 8380417  # 定義大質數 p
m = 512  # 定義矩陣行數
n = 256  # 定義矩陣列數
kyber_k = 3  # Kyber768中的參數k
kyber_n = 256  # Kyber768中的參數n
kyber_eta = 2  # Kyber768中的參數eta

# 生成公開矩陣 M ∈ Z_p^{m×n} 並轉換為 NTT 格式
def generate_NTT_matrix(m, n, p):
    M_real = np.random.randint(0, p, size=(m, n))  # 生成實部隨機矩陣
    M_imag = np.random.randint(0, p, size=(m, n))  # 生成虛部隨機矩陣
    M = M_real + 1j * M_imag  # 合併實部和虛部
    M_NTT = np.fft.fft(M, axis=0)  # 將矩陣轉換為NTT格式
    return M_NTT  # 返回NTT格式矩陣

M_NTT = generate_NTT_matrix(m, n, p)  # 生成並轉換公開矩陣

# 生成密鑰對
def generate_keypair():
    random.seed(time.time_ns())  # 每次生成密鑰前重設隨機數種子
    alpha = random.getrandbits(256)  # 生成隨機256位元的alpha
    s = np.random.randint(-kyber_eta, kyber_eta + 1, size=(kyber_n, kyber_k))  # 生成s
    s_NTT = np.fft.fft(s, axis=0)  # 將s轉換為NTT格式
    e = np.random.randint(-kyber_eta, kyber_eta + 1, size=(m, kyber_k))  # 生成e
    e_NTT = np.fft.fft(e, axis=0)  # 將e轉換為NTT格式

    # 分開處理實數和虛數部分的餘數計算
    t_real = np.remainder(np.dot(M_NTT.real, s_NTT.real) - np.dot(M_NTT.imag, s_NTT.imag) + e_NTT.real, p)
    t_imag = np.remainder(np.dot(M_NTT.real, s_NTT.imag) + np.dot(M_NTT.imag, s_NTT.real) + e_NTT.imag, p)
    t = t_real + 1j * t_imag

    pk = (t, alpha)  # 公鑰包含t和alpha
    sk = s_NTT  # 私鑰為s_NTT
    return pk, sk  # 返回公鑰和私鑰

participants = ['Alice', 'Bob', 'Charlie', 'Dave']  # 定義參與者
keys = {participant: generate_keypair() for participant in participants}  # 為每個參與者生成密鑰對

# FACKA密鑰生成
def facka_key_generation(salt, IKM):
    PRK = hmac.new(salt, IKM, sha256).digest()  # 生成伺服器密鑰
    OKM = hmac.new(PRK, b"info" + b'\\x01', sha256).digest()  # 生成輸出密鑰
    return PRK, OKM  # 返回伺服器密鑰和輸出密鑰

salt = b'random_salt'  # 定義隨機鹽值
IKM = b''.join([key[0][1].to_bytes(32, 'big') for key in keys.values()])  # 生成初始密鑰材料
PRK, OKM = facka_key_generation(salt, IKM)  # 生成FACKA密鑰

# 掩碼複數數值
def mask_complex(value, mask_value):
    real_masked = np.bitwise_xor(value.real.astype(np.int64), mask_value.astype(np.int64))  # 掩碼實部
    imag_masked = np.bitwise_xor(value.imag.astype(np.int64), mask_value.astype(np.int64))  # 掩碼虛部
    return real_masked + 1j * imag_masked  # 返回掩碼後的複數
"""
    hash_before = hashlib.sha256(code_content.encode()).digest()  # 計算原始程式碼的SHA256雜湊值
    return hash_before  # 返回雜湊值

# 驗證系統完整性
def verify_system_integrity(expected_hash):
    current_hash = check_system_integrity()  # 取得當前系統的雜湊值
    if expected_hash != current_hash:  # 比較預期雜湊值和當前雜湊值
        raise ValueError("System integrity check failed!")  # 如果不匹配，拋出錯誤

# 在加密和解密過程之前檢查系統完整性
system_integrity_hash = check_system_integrity()  # 計算初始系統完整性雜湊值

# 加密函數
def encrypt(message, pk, sk):
    verify_system_integrity(system_integrity_hash)  # 檢查系統完整性

    t, alpha = pk  # 提取公鑰中的t和alpha
    r = np.random.randint(-kyber_eta, kyber_eta + 1, size=(kyber_n, kyber_k))  # 生成隨機向量r
    r_NTT = np.fft.fft(r, axis=0)  # 將r轉換為NTT格式
    e1 = np.random.randint(-kyber_eta, kyber_eta + 1, size=(m, kyber_k))  # 生成誤差向量e1
    e1_NTT = np.fft.fft(e1, axis=0)  # 將e1轉換為NTT格式
    e2 = np.random.randint(-kyber_eta, kyber_eta + 1, size=(kyber_n,))  # 生成誤差向量e2

    mask_value = np.random.randint(0, 2**32 - 1)  # 生成掩碼值
    random_bytes = os.urandom(16)  # 生成隨機字節

    print(f"Encrypt mask_value: {mask_value}")  # 調試輸出掩碼值

    # 分開處理實數和虛數部分的餘數計算
    u_real = np.remainder(np.dot(M_NTT.real, r_NTT.real) - np.dot(M_NTT.imag, r_NTT.imag) + e1_NTT.real, p)
    u_imag = np.remainder(np.dot(M_NTT.real, r_NTT.imag) + np.dot(M_NTT.imag, r_NTT.real) + e1_NTT.imag, p)
    u = u_real + 1j * u_imag

    v_real = np.remainder(np.dot(t.real, r.real.T) - np.dot(t.imag, r.imag.T) + e2 + message, p)
    v_imag = np.remainder(np.dot(t.real, r.imag.T) + np.dot(t.imag, r.real.T), p)
    v = v_real + 1j * v_imag

    u_masked = mask_complex(u, mask_value)  # 對u進行掩碼
    v_masked = mask_complex(v, mask_value)  # 對v進行掩碼

    u_masked_bytes = u_masked.real.astype(np.int64).tobytes() + u_masked.imag.astype(np.int64).tobytes()  # 將掩碼後的u轉換為字節串
    v_masked_bytes = v_masked.real.astype(np.int64).tobytes() + v_masked.imag.astype(np.int64).tobytes()  # 將掩碼後的v轉換為字節串

    key = hashlib.sha256(sk.tobytes() + alpha.to_bytes(32, 'big') + random_bytes).digest()  # 基於私鑰和alpha生成隨機金鑰，加入隨機字節避免重複
    print(f"Encrypt key: {key.hex()}")  # 調試輸出密鑰

    hmac_val = hmac.new(key, u_masked_bytes + v_masked_bytes, sha256).digest()  # 計算HMAC值

    print(f"Encrypt HMAC: {hmac_val.hex()}")  # 調試輸出HMAC值

    return (u_masked, v_masked, hmac_val, mask_value, random_bytes)  # 返回加密後的值

votes = {'Alice': 1, 'Bob': 0, 'Charlie': 1, 'Dave': 0}  # 定義投票
encrypted_votes = {participant: encrypt(vote, keys[participant][0], keys[participant][1]) for participant, vote in votes.items()}  # 為每個參與者加密投票

# 解密函數
def decrypt(ciphertext, pk, sk):
    verify_system_integrity(system_integrity_hash)  # 檢查系統完整性

    t, alpha = pk  # 提取公鑰中的t和alpha
    u_masked, v_masked, received_hmac, mask_value, random_bytes = ciphertext  # 提取密文

    print(f"Decrypt mask_value: {mask_value}")  # 調試輸出掩碼值

    # 解密過程中不再對 u 和 v 進行掩碼操作
    u = u_masked
    v = v_masked

    u_bytes = u.real.astype(np.int64).tobytes() + u.imag.astype(np.int64).tobytes()  # 將u轉換為字節串
    v_bytes = v.real.astype(np.int64).tobytes() + v.imag.astype(np.int64).tobytes()  # 將v轉換為字節串

    key = hashlib.sha256(sk.tobytes() + alpha.to_bytes(32, 'big') + random_bytes).digest()  # 基於私鑰和alpha生成隨機金鑰
    print(f"Decrypt key: {key.hex()}")  # 調試輸出密鑰

    hmac_val = hmac.new(key, u_bytes + v_bytes, sha256).digest()  # 計算HMAC值

    print(f"Decrypt HMAC: {hmac_val.hex()}")  # 調試輸出HMAC值
    print(f"Received HMAC: {received_hmac.hex()}")  # 調試輸出接收到的HMAC值

    if not hmac.compare_digest(received_hmac, hmac_val):  # 比較HMAC值
        raise ValueError("HMAC verification failed!")  # 如果不匹配，拋出錯誤

    m_decoded_real = np.remainder(v.real - np.dot(u, sk.T.conj()).real, p)  # 計算解密後的實部
    print("Shape of u: ", u.shape)  # 調試輸出u的形狀
    print("Shape of sk.T.conj(): ", sk.T.conj().shape)  # 調試輸出私鑰的形狀
    print("Shape of np.dot(u, sk.T.conj()).real: ", np.dot(u, sk.T.conj()).real.shape)  # 調試輸出點積的形狀
    print("Shape of v.real: ", v.real.shape)  # 調試輸出v的形狀
    print("Shape of m_decoded_real: ", m_decoded_real.shape)  # 調試輸出解密後實部的形狀

    m_decoded_imag = np.remainder(v.imag - np.dot(u, sk.T.conj()).imag, p)  # 計算解密後的虛部
    m_decoded = m_decoded_real + 1j * m_decoded_imag  # 合併實部和虛部
    return m_decoded  # 返回解密後的值

decrypted_votes = {participant: decrypt(ciphertext, keys[participant][0], keys[participant][1]) for participant, ciphertext in encrypted_votes.items()}  # 為每個參與者解密投票
print(decrypted_votes)  # 輸出解密後的投票結果

# 模擬旁路攻擊
def simulate_side_channel_attack(encrypt_function, votes, keys):
    time_taken = []  # 初始化時間列表
    for i in range(200):  # 模擬200次加密操作
        start_time = time.time()  # 記錄開始時間
        encrypted_votes = {participant: encrypt_function(vote, keys[participant][0], keys[participant][1]) for participant, vote in votes.items()}  # 為每個參與者加密投票
        end_time = time.time()  # 記錄結束時間
        time_taken.append(end_time - start_time)  # 計算並記錄加密時間
    return time_taken  # 返回時間列表

# 模擬旁路攻擊
side_channel_times = simulate_side_channel_attack(encrypt, votes, keys)  # 執行旁路攻擊模擬

# 畫出加密運算時間圖
plt.figure(figsize=(10, 5))  # 設置圖形大小
plt.plot(side_channel_times, label='Encryption Time')  # 繪製加密時間圖
plt.xlabel('Simulation Iteration')  # 設置X軸標籤
plt.ylabel('Time (seconds)')  # 設置Y軸標籤
plt.title('Side-channel Attack Simulation: Encryption Time Analysis')  # 設置圖形標題
plt.legend()  # 顯示圖例
plt.grid(True)  # 顯示網格
plt.show()  # 顯示圖形
