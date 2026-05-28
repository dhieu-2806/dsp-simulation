import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Cài đặt giao diện trang web
st.set_page_config(page_title="Mô phỏng DSP", layout="wide")
st.title("Mô phỏng DSP: Lấy mẫu & Phổ Tần Số")

# ==========================================
# 1. HÀM TÍNH TOÁN LÕI (Giữ nguyên của bạn)
# ==========================================
# Dùng @st.cache_data để tối ưu tốc độ tính toán lại nếu thông số không đổi
@st.cache_data
def calculate_signals(f_sig, f_samp, wave_type):
    t_cont = np.linspace(0, 1, 1000)
    
    if wave_type == 'Sin':
        x_cont = np.cos(2 * np.pi * f_sig * t_cont)
    elif wave_type == 'Vuông':
        x_cont = signal.square(2 * np.pi * f_sig * t_cont)
    else: 
        x_cont = signal.sawtooth(2 * np.pi * f_sig * t_cont, 0.5)
    
    T_s = 1.0 / f_samp if f_samp > 0 else 1.0
    t_disc = np.arange(0, 1 + T_s, T_s)
    
    if wave_type == 'Sin':
        x_disc = np.cos(2 * np.pi * f_sig * t_disc)
    elif wave_type == 'Vuông':
        x_disc = signal.square(2 * np.pi * f_sig * t_disc)
    else:
        x_disc = signal.sawtooth(2 * np.pi * f_sig * t_disc, 0.5)
    
    x_recon = np.zeros_like(t_cont)
    for n, x_n in enumerate(x_disc):
        nT = t_disc[n]
        x_recon += x_n * np.sinc((t_cont - nT) / T_s)
        
    N = len(x_disc)
    freqs = np.fft.fftfreq(N, T_s)
    fft_vals = np.abs(np.fft.fft(x_disc)) / N
    
    pos_mask = freqs >= 0
    
    return t_cont, x_cont, t_disc, x_disc, x_recon, freqs[pos_mask], fft_vals[pos_mask]

# ==========================================
# 2. KHU VỰC ĐIỀU KHIỂN (SIDEBAR)
# ==========================================
st.sidebar.header("Thông số cài đặt")
wave_type = st.sidebar.radio('Loại sóng', ('Sin', 'Vuông', 'Tam giác'))
f_sig = st.sidebar.slider('Tần số tín hiệu (Hz)', 1.0, 20.0, 2.0, step=1.0)
f_samp = st.sidebar.slider('Tần số lấy mẫu (Hz)', 2.0, 100.0, 20.0, step=1.0)

# ==========================================
# 3. GỌI HÀM TÍNH TOÁN
# ==========================================
t_c, x_c, t_d, x_d, x_r, freqs, fft_vals = calculate_signals(f_sig, f_samp, wave_type)

# ==========================================
# 4. VẼ BIỂU ĐỒ BẰNG MATPLOTLIB
# ==========================================
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 12))
plt.subplots_adjust(hspace=0.5) 

# Biểu đồ 1
ax1.plot(t_c, x_c, 'b-', linewidth=2)
ax1.set_title('1. Tín hiệu gốc liên tục: $x(t)$', fontweight='bold')
ax1.set_xlim(0, 1)
ax1.set_ylim(-1.5, 1.5)
ax1.grid(True)

# Biểu đồ 2
ax2.stem(t_d, x_d, linefmt='r-', markerfmt='ro', basefmt='k-')
ax2.set_title('2. Tín hiệu sau lấy mẫu: $x(nT)$', fontweight='bold')
ax2.set_xlim(0, 1)
ax2.set_ylim(-1.5, 1.5)
ax2.grid(True)

# Biểu đồ 3
ax3.plot(t_c, x_r, 'g-', linewidth=2)
ax3.set_title('3. Tín hiệu khôi phục lý tưởng: $x_r(t)$', fontweight='bold')
ax3.set_xlim(0, 1)
ax3.set_ylim(-1.5, 1.5)
ax3.grid(True)

# Biểu đồ 4
ax4.stem(freqs, fft_vals, linefmt='m-', markerfmt='mo', basefmt='k-')
ax4.set_title('4. Phổ tần số của tín hiệu lấy mẫu (FFT)', fontweight='bold')
ax4.set_xlim(-1, max(f_samp, f_sig * 3))
ax4.set_ylim(0, 1.2)
ax4.set_xlabel('Tần số (Hz)')
ax4.set_ylabel('Biên độ')
ax4.grid(True)

# ==========================================
# 5. HIỂN THỊ LÊN TRÌNH DUYỆT
# ==========================================
st.pyplot(fig)