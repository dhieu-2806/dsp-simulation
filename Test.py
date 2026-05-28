import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import pandas as pd

# ==========================================
# 1. CẤU HÌNH & GIAO DIỆN CHUNG
# ==========================================
st.set_page_config(page_title="Mô phỏng lấy mẫu", page_icon="🎛️", layout="wide")
st.markdown("<h1 style='text-align: center; margin-bottom: 0;'>🎛️ Mô phỏng lấy mẫu và khôi phục</h1><hr>", unsafe_allow_html=True)

with st.expander("📖 Xem giải thích lý thuyết & Công thức"):
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown("**Nyquist:**\n$f_s \\ge 2 f_{max}$")
    c2.markdown("**Nội suy Sinc:**\n$x_r(t) = \\sum x(nT_s)\\cdot \\mathrm{sinc}(\\frac{t - nT_s}{T_s})$")
    c3.markdown("**FFT:**\n$X[k] = \\sum x[n]\\, e^{-j2\\pi kn/N}$")
    c4.markdown("**SNR:**\n$\\mathrm{SNR} = 10\\log_{10}(\\frac{\\sum x^2}{\\sum (x-x_r)^2})$")

# ==========================================
# 2. HÀM TÍNH TOÁN LÕI 
# ==========================================
@st.cache_data
def calculate_signals(f_sig, f_samp, wave_type):
    t_cont = np.linspace(0, 1, 2000)
    T_s = 1.0 / f_samp
    
    # Boundary padding để tránh méo biên
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
    st.markdown("### ⚙️ Bảng Điều Khiển")
    wave_type = st.radio("1. Chọn loại sóng", ("Sin", "Vuông", "Tam giác"))
    f_sig  = st.number_input("2. Tần số tín hiệu (Hz)", 1.0, 50.0, 2.0, 0.5)
    f_samp = st.number_input("3. Tần số lấy mẫu (Hz)", 2.0, 300.0, 20.0, 1.0) # Đã giới hạn max_value để chống treo

t_c, x_c, t_d, x_d, x_r, err, mse, snr, f_pos, fft_pos, f_two, fft_two = calculate_signals(f_sig, f_samp, wave_type)

f_nyq = f_samp / 2
req_nyq = 2 * f_sig
alias = f_samp < req_nyq

cols = st.columns(4)
cols[0].metric("Tần số tín hiệu", f"{f_sig} Hz")
cols[1].metric("Tần số lấy mẫu", f"{f_samp} Hz")
cols[2].metric("Nyquist yêu cầu", f"{req_nyq} Hz", f"{f_samp - req_nyq:+.1f} Hz", "normal" if not alias else "inverse")
cols[3].metric("SNR khôi phục", f"{snr:.1f} dB", "Tốt" if snr > 20 else "Kém", "normal" if snr > 20 else "inverse")

if not alias: st.success(f"✅ **THỎA MÃN NYQUIST** — Không xảy ra Aliasing. SNR = {snr:.1f} dB.")
else: st.error(f"🚨 **ALIASING NGHIÊM TRỌNG** — Tín hiệu bị biến dạng. SNR = {snr:.1f} dB.")

# ==========================================
# 4. BIỂU ĐỒ (Dùng Helper Function để tránh lặp code)
# ==========================================
def create_plot(plot_func, title, xlabel="Thời gian (s)", ylabel="Biên độ", xlim=(0, 1), ylim=(-1.6, 1.6)):
    fig, ax = plt.subplots(figsize=(10, 3.5))
    plot_func(ax)
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel, xlim=xlim, ylim=ylim)
    ax.grid(True, ls='--', alpha=0.4)
    ax.spines[['top', 'right']].set_visible(False)
    if ax.get_legend_handles_labels()[0]: ax.legend(fontsize=9, loc='upper right')
    return fig

tabs = st.tabs(["📈 Gốc vs Khôi phục", "📍 Lấy mẫu", "📉 Sai số & SNR", "📊 Phổ 1 phía", "🔁 Phổ 2 phía"])

with tabs[0]:
    st.pyplot(create_plot(lambda ax: [
        ax.plot(t_c, x_c, '#2E86AB', lw=2.5, label='Gốc $x(t)$', zorder=3),
        ax.plot(t_c, x_r, '#388659', lw=2, ls='--', label='Khôi phục $x_r(t)$', zorder=2),
        ax.scatter(t_d, x_d, c='#D63230', s=40, label='Mẫu', zorder=4)
    ], "So sánh tín hiệu gốc và khôi phục"))

with tabs[1]:
    st.pyplot(create_plot(lambda ax: [
        ax.plot(t_c, x_c, '#2E86AB', lw=1.2, alpha=0.35, label='Gốc'),
        ax.stem(t_d, x_d, linefmt='r-', markerfmt='ro', basefmt='k-', label='Mẫu rời rạc')
    ], f"Tín hiệu sau lấy mẫu ({len(t_d)} mẫu)"))

with tabs[2]:
    c1, c2 = st.columns([1.5, 1])
    with c1:
        st.pyplot(create_plot(lambda ax: [
            ax.fill_between(t_c, err, color='#E07A5F', alpha=0.6, label='Sai số $e(t)$'),
            ax.axhline(0, color='k', lw=0.8)
        ], f"Sai số khôi phục", ylim=(-max(abs(err))-0.1, max(abs(err))+0.1)))
    with c2:
        st.markdown("### Chỉ số thống kê")
        df_stats = pd.DataFrame([
            {"Chỉ số": "MSE", "Giá trị": f"{mse:.6f}"},
            {"Chỉ số": "SNR", "Giá trị": f"{snr:.2f} dB"},
            {"Chỉ số": "Lỗi cực đại", "Giá trị": f"{np.max(np.abs(err)):.4f}"}
        ])
        st.dataframe(df_stats, use_container_width=True, hide_index=True)

with tabs[3]:
    st.pyplot(create_plot(lambda ax: [
        ax.stem(f_pos, fft_pos, linefmt='m-', markerfmt='mo', basefmt='k-', label='FFT'),
        ax.axvline(f_nyq, color='r', ls='--', lw=2, label=f'Nyquist ({f_nyq:.1f} Hz)'),
        ax.axvline(f_sig, color='b', ls=':', lw=1.5, label=f'f_sig ({f_sig} Hz)')
    ], "Phổ một phía", "Tần số (Hz)", "Biên độ", (-0.5, max(f_samp * 0.6, f_sig * 3)), (0, 1.2)))

with tabs[4]:
    st.pyplot(create_plot(lambda ax: [
        ax.stem(f_two, fft_two, linefmt='b-', markerfmt='bo', basefmt='k-', label='FFT'),
        ax.axvline(f_nyq, color='r', ls='--', lw=2, label=f'+Nyq ({f_nyq:.1f} Hz)'),
        ax.axvline(-f_nyq, color='r', ls='--', lw=2, label=f'-Nyq ({-f_nyq:.1f} Hz)'),
        ax.axvspan(-f_nyq, f_nyq, alpha=0.06, color='g', label='Băng thông')
    ], "Phổ hai phía (Gập méo khi Aliasing)", "Tần số (Hz)", "Biên độ", (-f_samp * 0.6, f_samp * 0.6), (0, 1.2)))

# ==========================================
# 5. XUẤT DỮ LIỆU
# ==========================================
st.markdown("---")
c1, c2 = st.columns(2)
c1.download_button("📥 Tải Phổ FFT (.CSV)", pd.DataFrame({'Tần số': f_pos, 'Biên độ': fft_pos}).to_csv(index=False), 'fft.csv', 'text/csv')
c2.download_button("📥 Tải Sai số (.CSV)", pd.DataFrame({'Thời gian': t_c, 'Gốc': x_c, 'Khôi phục': x_r, 'Sai số': err}).to_csv(index=False), 'sai_so.csv', 'text/csv')
