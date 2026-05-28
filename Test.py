import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# ==========================================
# CẤU HÌNH TRANG WEB
# ==========================================
st.set_page_config(page_title="Đồ án Mô phỏng DSP", page_icon="🎛️", layout="wide")

# Header chuyên nghiệp
st.markdown("<h4 style='text-align: center; color: #154c79;'>ĐẠI HỌC BÁCH KHOA ĐHQG-HCM</h4>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center;'>🎛️ Đồ án DSP: Lấy mẫu & Khôi phục Tín hiệu</h1>", unsafe_allow_html=True)
st.markdown("---")

# ==========================================
# KHU VỰC LÝ THUYẾT & CÔNG THỨC (Gập/Mở)
# ==========================================
with st.expander("📖 Xem giải thích lý thuyết & Công thức toán học"):
    st.markdown("**1. Định lý lấy mẫu Nyquist-Shannon:**")
    st.markdown("Để khôi phục lại tín hiệu gốc mà không bị hiện tượng chồng phổ (Aliasing), tần số lấy mẫu ($f_s$) phải lớn hơn hoặc bằng hai lần tần số cực đại của tín hiệu ($f_{max}$).")
    st.latex(r"f_s \ge 2f_{max}")
    
    st.markdown("**2. Khôi phục tín hiệu lý tưởng (Sinc Interpolation):**")
    st.markdown("Tín hiệu liên tục được khôi phục từ các mẫu rời rạc bằng cách chập chuỗi xung mẫu với hàm Sinc lý tưởng.")
    st.latex(r"x_r(t) = \sum_{n=-\infty}^{\infty} x(nT) \cdot \text{sinc}\left(\frac{t - nT}{T_s}\right)")

# ==========================================
# HÀM TÍNH TOÁN LÕI 
# ==========================================
@st.cache_data
def calculate_signals(f_sig, f_samp, wave_type):
    # Tự động điều chỉnh số lượng điểm lấy mẫu đồ họa để đường nét luôn mịn ở tần số cao
    num_points = int(max(1000, 20 * max(f_sig, f_samp)))
    t_cont = np.linspace(0, 1, num_points)
    
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
# ĐIỀU KHIỂN (SIDEBAR) VỚI NUMBER INPUT
# ==========================================
st.sidebar.markdown("### ⚙️ Bảng Điều Khiển")
wave_type = st.sidebar.radio('1. Chọn loại sóng', ('Sin', 'Vuông', 'Tam giác'))

# Nâng cấp lên Number Input để nhập số chính xác tuyệt đối
f_sig = st.sidebar.number_input('2. Tần số tín hiệu (Hz)', min_value=1.0, max_value=5000.0, value=2.0, step=0.5, format="%.1f")
f_samp = st.sidebar.number_input('3. Tần số lấy mẫu (Hz)', min_value=2.0, max_value=10000.0, value=20.0, step=1.0, format="%.1f")

t_c, x_c, t_d, x_d, x_r, freqs, fft_vals = calculate_signals(f_sig, f_samp, wave_type)

# ==========================================
# VẼ BIỂU ĐỒ 
# ==========================================
COLOR_ORIGINAL = '#2E86AB'  
COLOR_RECON = '#388659'     

fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 14))
plt.subplots_adjust(hspace=0.6) 

# Cấu hình chung cho lưới (Grid)
for ax in (ax1, ax2, ax3, ax4):
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# Biểu đồ 1
ax1.plot(t_c, x_c, color=COLOR_ORIGINAL, linewidth=2.5)
ax1.set_title('1. Tín hiệu gốc liên tục: $x(t)$', fontweight='bold', color=COLOR_ORIGINAL)
ax1.set_xlim(0, 1)
ax1.set_ylim(-1.5, 1.5)

# Biểu đồ 2 
ax2.stem(t_d, x_d, linefmt='r-', markerfmt='ro', basefmt='k-')
ax2.set_title('2. Tín hiệu sau lấy mẫu: $x(nT)$', fontweight='bold', color='#D63230')
ax2.set_xlim(0, 1)
ax2.set_ylim(-1.5, 1.5)

# Biểu đồ 3
ax3.plot(t_c, x_r, color=COLOR_RECON, linewidth=2.5)
ax3.set_title('3. Tín hiệu khôi phục lý tưởng (Nội suy Sinc): $x_r(t)$', fontweight='bold', color=COLOR_RECON)
ax3.set_xlim(0, 1)
ax3.set_ylim(-1.5, 1.5)

# Biểu đồ 4 
ax4.stem(freqs, fft_vals, linefmt='m-', markerfmt='mo', basefmt='k-')
ax4.set_title('4. Phổ tần số của tín hiệu lấy mẫu (FFT)', fontweight='bold', color='#8A4F7D')
ax4.set_xlim(-1, max(f_samp, f_sig * 3))
ax4.set_ylim(0, 1.2)
ax4.set_xlabel('Tần số (Hz)', fontstyle='italic')
ax4.set_ylabel('Biên độ', fontstyle='italic')

st.pyplot(fig)