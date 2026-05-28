import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import pandas as pd

# ==========================================
# 1. CẤU HÌNH & GIAO DIỆN CHUNG
# ==========================================
st.set_page_config(page_title="Mô phỏng lấy mẫu", layout="wide")
st.markdown("<h2 style='text-align: center;'>Mô phỏng Lấy mẫu và Khôi phục Tín hiệu</h2>", unsafe_allow_html=True)
st.divider()

# ==========================================
# 2. HÀM TÍNH TOÁN LÕI 
# ==========================================
@st.cache_data
def calculate_signals(f_sig, f_samp, wave_type):
    t_cont = np.linspace(0, 1, 2000)
    T_s = 1.0 / f_samp
    
    pad = min(int(f_samp), 50)
    t_disc_pad = np.arange(-pad * T_s, 1 + (pad + 1) * T_s, T_s)
    t_disc = np.arange(0, 1 + T_s, T_s)

    def gen_wave(t):
        if wave_type == 'Sin': return np.cos(2 * np.pi * f_sig * t)
        if wave_type == 'Vuông': return signal.square(2 * np.pi * f_sig * t)
        return signal.sawtooth(2 * np.pi * f_sig * t, 0.5)

    x_cont, x_disc_pad, x_disc = gen_wave(t_cont), gen_wave(t_disc_pad), gen_wave(t_disc)

    # Vectorized Sinc Interpolation
    x_recon = x_disc_pad @ np.sinc((t_cont[:, None] - t_disc_pad[None, :]) / T_s).T

    # Sai số & SNR
    error = x_cont - x_recon
    mse = np.mean(error**2)
    snr_db = 10 * np.log10(np.mean(x_cont**2) / (mse + 1e-12))

    # Phổ FFT
    N = len(x_disc)
    freqs = np.fft.fftfreq(N, T_s)
    fft_amps = np.abs(np.fft.fft(x_disc)) / N
    pos_mask = freqs >= 0

    return t_cont, x_cont, t_disc, x_disc, x_recon, error, mse, snr_db, freqs[pos_mask], fft_amps[pos_mask], np.fft.fftshift(freqs), np.fft.fftshift(fft_amps)

# ==========================================
# 3. SIDEBAR & METRICS
# ==========================================
with st.sidebar:
    st.markdown("### 🎛️ Bảng Điều Khiển")
    wave_type = st.radio("1. Chọn loại sóng", ("Sin", "Vuông", "Tam giác"))
    f_sig  = st.number_input("2. Tần số tín hiệu (Hz)", 1.0, 50.0, 2.0, 0.5)
    f_samp = st.number_input("3. Tần số lấy mẫu (Hz)", 2.0, 300.0, 20.0, 1.0)

t_c, x_c, t_d, x_d, x_r, err, mse, snr, f_pos, fft_pos, f_two, fft_two = calculate_signals(f_sig, f_samp, wave_type)

f_nyq = f_samp / 2
req_nyq = 2 * f_sig
alias = f_samp < req_nyq

# Sử dụng container có viền để đóng khung các metric
with st.container(border=True):
    cols = st.columns(4)
    cols[0].metric("Tần số tín hiệu", f"{f_sig} Hz")
    cols[1].metric("Tần số lấy mẫu", f"{f_samp} Hz")
    cols[2].metric("Nyquist yêu cầu", f"{req_nyq} Hz", f"{f_samp - req_nyq:+.1f} Hz", "normal" if not alias else "inverse")
    cols[3].metric("SNR khôi phục", f"{snr:.1f} dB", "Tốt" if snr > 20 else "Kém", "normal" if snr > 20 else "inverse")

# Hiển thị thông báo với giao diện điềm đạm hơn
if not alias: 
    st.info(f"**THỎA MÃN NYQUIST** — Không xảy ra hiện tượng chồng phổ. SNR = {snr:.1f} dB.", icon="✅")
else: 
    st.warning(f"**CẢNH BÁO ALIASING** — Tín hiệu bị biến dạng. SNR = {snr:.1f} dB.", icon="⚠️")

# ==========================================
# 4. BIỂU ĐỒ & TRỰC QUAN HÓA
# ==========================================
def create_plot(plot_func, title, xlabel="Thời gian (s)", ylabel="Biên độ", xlim=(0, 1), ylim=(-1.6, 1.6)):
    # Đồng bộ giao diện Dark Mode
    plt.style.use('dark_background')
    
    fig, ax = plt.subplots(figsize=(12, 3.5))
    
    # Nền trong suốt
    fig.patch.set_alpha(0.0)
    ax.set_facecolor((0, 0, 0, 0))
    
    plot_func(ax)
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel, xlim=xlim, ylim=ylim)
    
    # Lưới tinh tế hơn
    ax.grid(True, ls=':', alpha=0.3, color='#AAAAAA')
    
    # Bỏ viền thừa
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('#777777')
    ax.spines['left'].set_color('#777777')
    
    if ax.get_legend_handles_labels()[0]: 
        ax.legend(fontsize=9, loc='best', framealpha=0.2, edgecolor='#555')
    
    fig.tight_layout()
    return fig

tabs = st.tabs(["Gốc vs Khôi phục", "Lấy mẫu", "Sai số & SNR", "Phổ 1 phía", "Phổ 2 phía"])

# Cần truyền thêm tham số transparent=True vào st.pyplot
with tabs[0]:
    st.pyplot(create_plot(lambda ax: [
        ax.plot(t_c, x_c, '#2E86AB', lw=2.5, label='Tin hieu goc', zorder=3),
        ax.plot(t_c, x_r, '#388659', lw=2, ls='--', label='Khoi phuc', zorder=2),
        ax.scatter(t_d, x_d, c='#D63230', s=40, label='Mau roi rac', zorder=4)
    ], "So sánh tín hiệu gốc và tín hiệu khôi phục"), transparent=True)

with tabs[1]:
    st.pyplot(create_plot(lambda ax: [
        ax.plot(t_c, x_c, '#2E86AB', lw=1.2, alpha=0.35, label='Tham chieu goc'),
        ax.stem(t_d, x_d, linefmt='r-', markerfmt='ro', basefmt='k-', label='Gia tri lay mau')
    ], f"Chuỗi tín hiệu sau khi lấy mẫu ({len(t_d)} điểm)"), transparent=True)

with tabs[2]:
    st.pyplot(create_plot(lambda ax: [
        ax.fill_between(t_c, err, color='#E07A5F', alpha=0.6, label='Sai so e(t)'),
        ax.axhline(0, color='k', lw=0.8)
    ], "Đồ thị sai số tái tạo theo thời gian", ylim=(-max(abs(err))-0.05, max(abs(err))+0.05)), transparent=True)
    
    df_stats = pd.DataFrame([
        {"Đại lượng": "Mean Squared Error (MSE)", "Giá trị": f"{mse:.6f}"},
        {"Đại lượng": "Signal-to-Noise Ratio (SNR)", "Giá trị": f"{snr:.2f} dB"},
        {"Đại lượng": "Sai số cực đại (Max Error)", "Giá trị": f"{np.max(np.abs(err)):.4f}"}
    ])
    st.table(df_stats)

with tabs[3]:
    st.pyplot(create_plot(lambda ax: [
        ax.stem(f_pos, fft_pos, linefmt='m-', markerfmt='mo', basefmt='k-', label='Pho FFT'),
        ax.axvline(f_nyq, color='r', ls='--', lw=2, label=f'Gioi han Nyquist ({f_nyq:.1f} Hz)'),
        ax.axvline(f_sig, color='b', ls=':', lw=1.5, label=f'Tan so goc ({f_sig} Hz)')
    ], "Phổ biên độ một phía", "Tần số (Hz)", "Biên độ", xlim=(0, f_samp), ylim=(0, 1.2)), transparent=True)

with tabs[4]:
    st.pyplot(create_plot(lambda ax: [
        ax.stem(f_two, fft_two, linefmt='b-', markerfmt='bo', basefmt='k-', label='Pho FFT'),
        ax.axvline(f_nyq, color='r', ls='--', lw=2, label=f'+Nyquist ({f_nyq:.1f} Hz)'),
        ax.axvline(-f_nyq, color='r', ls='--', lw=2, label=f'-Nyquist ({-f_nyq:.1f} Hz)'),
        ax.axvspan(-f_nyq, f_nyq, alpha=0.1, color='green', label='Vung Baseband')
    ], "Phổ biên độ hai phía (Quan sát hiện tượng gập phổ)", "Tần số (Hz)", "Biên độ", xlim=(-f_samp, f_samp), ylim=(0, 1.2)), transparent=True)

# ==========================================
# 5. XUẤT DỮ LIỆU
# ==========================================
st.divider()
c1, c2 = st.columns(2)
c1.download_button("Tải dữ liệu Phổ FFT (.CSV)", pd.DataFrame({'Tan so': f_pos, 'Bien do': fft_pos}).to_csv(index=False), 'fft_data.csv', 'text/csv')
c2.download_button("Tải dữ liệu Sai số (.CSV)", pd.DataFrame({'Thoi gian': t_c, 'Goc': x_c, 'Khoi phuc': x_r, 'Sai so': err}).to_csv(index=False), 'error_data.csv', 'text/csv')
