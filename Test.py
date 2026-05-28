import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.interpolate import interp1d
import pandas as pd

# ==========================================
# 1. CẤU HÌNH & GIAO DIỆN CHUNG
# ==========================================
st.set_page_config(page_title="Mô phỏng lấy mẫu DSP", layout="wide")
st.markdown("<h2 style='text-align: center;'>Mô phỏng Hệ thống Lấy mẫu và Khôi phục DSP</h2>", unsafe_allow_html=True)
st.divider()

# ==========================================
# 2. HÀM TÍNH TOÁN LÕI 
# ==========================================
@st.cache_data
def calculate_signals(f_sig, f_samp, wave_type, use_lpf, recon_method):
    # Mô phỏng thời gian liên tục với độ phân giải cao (F_sim = 5000 Hz)
    F_sim = 5000
    t_cont = np.linspace(0, 1, F_sim)
    T_s = 1.0 / f_samp
    
    def gen_wave(t):
        if wave_type == 'Sin': return np.cos(2 * np.pi * f_sig * t)
        if wave_type == 'Vuông': return signal.square(2 * np.pi * f_sig * t)
        return signal.sawtooth(2 * np.pi * f_sig * t, 0.5)

    x_cont_raw = gen_wave(t_cont)

    # Khối Anti-aliasing Filter (Low-Pass Filter trước lấy mẫu)
    if use_lpf:
        nyq_sim = F_sim / 2
        cutoff = f_samp / 2
        if cutoff < nyq_sim:
            b, a = signal.butter(4, cutoff / nyq_sim, btype='low')
            x_cont = signal.filtfilt(b, a, x_cont_raw)
        else:
            x_cont = x_cont_raw
    else:
        x_cont = x_cont_raw

    # Quá trình lấy mẫu
    pad = min(int(f_samp), 50)
    t_disc_pad = np.arange(-pad * T_s, 1 + (pad + 1) * T_s, T_s)
    t_disc = np.arange(0, 1 + T_s, T_s)

    if use_lpf:
        x_disc_pad = np.interp(t_disc_pad, t_cont, x_cont)
        x_disc = np.interp(t_disc, t_cont, x_cont)
    else:
        x_disc_pad = gen_wave(t_disc_pad)
        x_disc = gen_wave(t_disc)

    # Khối Khôi phục (Reconstruction)
    if recon_method == 'Sinc (Lý tưởng)':
        # Sửa lại chiều transpose ma trận Sinc
        sinc_matrix = np.sinc((t_cont - t_disc_pad[:, None]) / T_s)
        x_recon = x_disc_pad @ sinc_matrix
    elif recon_method == 'Zero-Order Hold (ZOH)':
        f_zoh = interp1d(t_disc_pad, x_disc_pad, kind='previous', bounds_error=False, fill_value="extrapolate")
        x_recon = f_zoh(t_cont)
    else: # Linear
        f_lin = interp1d(t_disc_pad, x_disc_pad, kind='linear', bounds_error=False, fill_value="extrapolate")
        x_recon = f_lin(t_cont)

    # Tính toán lỗi
    error = x_cont - x_recon
    mse = np.mean(error**2)
    snr_db = 10 * np.log10(np.mean(x_cont**2) / (mse + 1e-12))

    # Xử lý phổ FFT
    N = len(x_disc)
    freqs = np.fft.fftfreq(N, T_s)
    fft_amps = np.abs(np.fft.fft(x_disc)) / N
    
    pos_mask = freqs >= 0
    f_pos = freqs[pos_mask]
    fft_pos = np.copy(fft_amps[pos_mask])
    
    # Nhân đôi biên độ cho các hài AC (Trừ thành phần DC tại index 0)
    if len(fft_pos) > 1:
        fft_pos[1:] = fft_pos[1:] * 2

    return t_cont, x_cont, t_disc, x_disc, x_recon, error, mse, snr_db, f_pos, fft_pos, np.fft.fftshift(freqs), np.fft.fftshift(fft_amps)

# ==========================================
# 3. SIDEBAR & BẢNG ĐIỀU KHIỂN
# ==========================================
with st.sidebar:
    st.markdown("### 🎛️ Cấu hình hệ thống")
    wave_type = st.radio("1. Nguồn phát (Tín hiệu)", ("Sin", "Vuông", "Tam giác"))
    f_sig  = st.number_input("2. Tần số cơ bản (Hz)", 1.0, 100.0, 5.0, 1.0)
    
    st.markdown("---")
    use_lpf = st.toggle("🛡️ Bật LPF (Anti-aliasing)", value=False)
    f_samp = st.number_input("3. Tần số lấy mẫu (Hz)", 2.0, 300.0, 8.0, 1.0)
    
    st.markdown("---")
    recon_method = st.selectbox("4. Mạch khôi phục", ("Sinc (Lý tưởng)", "Zero-Order Hold (ZOH)", "Nội suy tuyến tính"))

t_c, x_c, t_d, x_d, x_r, err, mse, snr, f_pos, fft_pos, f_two, fft_two = calculate_signals(f_sig, f_samp, wave_type, use_lpf, recon_method)

# Phân tích Nyquist & Aliasing
f_nyq = f_samp / 2
req_nyq = 2 * f_sig
alias = f_samp < req_nyq

# Tần số gập (Aliased Frequency) đối với hài cơ bản
n_alias = round(f_sig / f_samp)
f_alias = abs(f_sig - n_alias * f_samp)

with st.container(border=True):
    cols = st.columns(4)
    cols[0].metric("Tần số tín hiệu", f"{f_sig} Hz")
    cols[1].metric("Tần số lấy mẫu", f"{f_samp} Hz")
    cols[2].metric("Giới hạn Nyquist", f"{f_nyq} Hz", f"{f_nyq - f_sig:+.1f} Hz so với fs", "normal" if not alias else "inverse")
    cols[3].metric("SNR khôi phục", f"{snr:.1f} dB", f"Phương pháp: {recon_method[:4]}")

# Hệ thống cảnh báo thông minh
if wave_type != 'Sin' and not use_lpf:
    st.warning("⚠️ **ĐẶC TÍNH PHỔ:** Sóng Vuông/Tam giác có dải phổ rộng vô hạn (infinite harmonics). Nếu không bật LPF, các hài bậc cao chắc chắn sẽ gây ra hiện tượng chồng phổ (Aliasing) làm biến dạng tín hiệu, bất kể tần số lấy mẫu là bao nhiêu.")

if not alias: 
    st.info(f"✅ **HÀI CƠ BẢN THỎA MÃN NYQUIST** — Hệ thống có thể khôi phục thành phần {f_sig} Hz.")
else: 
    st.error(f"🚨 **CHỒNG PHỔ HÀI CƠ BẢN (ALIASING)** — Tần số {f_sig} Hz bị gập về **{f_alias:.1f} Hz** trong miền số.")

# ==========================================
# 4. TRỰC QUAN HÓA MATPLOTLIB
# ==========================================
def create_plot(plot_func, title, xlabel="Thời gian (s)", ylabel="Biên độ", xlim=(0, 1), ylim=(-1.6, 1.6)):
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 3.5))
    fig.patch.set_alpha(0.0)
    ax.set_facecolor((0, 0, 0, 0))
    plot_func(ax)
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel, xlim=xlim, ylim=ylim)
    ax.grid(True, ls=':', alpha=0.3, color='#AAAAAA')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('#777777')
    ax.spines['left'].set_color('#777777')
    
    if ax.get_legend_handles_labels()[0]: 
        ax.legend(fontsize=9, bbox_to_anchor=(0.5, 1.15), loc='center', ncol=3, framealpha=0.2, edgecolor='#555')
    
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig

tabs = st.tabs(["Gốc vs Khôi phục", "Sai số Tái tạo", "Phổ 1 phía (Phân tích Aliasing)", "Phổ 2 phía"])

with tabs[0]:
    st.pyplot(create_plot(lambda ax: [
        ax.plot(t_c, x_c, '#2E86AB', lw=2.5, label='Tín hiệu gốc (Đã qua LPF)' if use_lpf else 'Tín hiệu gốc', zorder=3),
        ax.plot(t_c, x_r, '#388659', lw=2, ls='--', label='Tín hiệu khôi phục', zorder=2),
        ax.scatter(t_d, x_d, c='#D63230', s=40, label='Mẫu rời rạc', zorder=4)
    ], "So sánh tín hiệu gốc và tín hiệu khôi phục"), transparent=True)

with tabs[1]:
    st.pyplot(create_plot(lambda ax: [
        ax.fill_between(t_c, err, color='#E07A5F', alpha=0.6, label='Sai số biên độ e(t)'),
        ax.axhline(0, color='k', lw=0.8)
    ], f"Đồ thị sai số theo thời gian (MSE = {mse:.4f})", ylim=(-max(abs(err))-0.1, max(abs(err))+0.1)), transparent=True)

with tabs[2]:
    st.pyplot(create_plot(lambda ax: [
        ax.stem(f_pos, fft_pos, linefmt='m-', markerfmt='mo', basefmt='k-', label='Phổ tín hiệu số'),
        ax.axvline(f_nyq, color='r', ls='-', lw=2, label=f'Nyquist ({f_nyq:.1f} Hz)'),
        ax.axvline(f_sig, color='b', ls=':', lw=1.5, label=f'Hài cơ bản ({f_sig} Hz)'),
        ax.axvline(f_alias, color='y', ls='--', lw=2, label=f'Tần số gập ({f_alias:.1f} Hz)') if alias else None
    ], "Phổ biên độ một phía & Hiện tượng chồng phổ", "Tần số (Hz)", "Biên độ", xlim=(0, max(f_samp, f_sig + 5)), ylim=(0, 1.2)), transparent=True)

with tabs[3]:
    st.pyplot(create_plot(lambda ax: [
        ax.stem(f_two, fft_two, linefmt='b-', markerfmt='bo', basefmt='k-', label='Phổ FFT 2 phía'),
        ax.axvline(f_nyq, color='r', ls='--', lw=2, label=f'+Nyquist ({f_nyq:.1f} Hz)'),
        ax.axvline(-f_nyq, color='r', ls='--', lw=2, label=f'-Nyquist ({-f_nyq:.1f} Hz)'),
        ax.axvspan(-f_nyq, f_nyq, alpha=0.1, color='green', label='Vùng Baseband')
    ], "Phổ biên độ hai phía", "Tần số (Hz)", "Biên độ", xlim=(-max(f_samp, f_sig + 5), max(f_samp, f_sig + 5)), ylim=(0, 1.2)), transparent=True)
