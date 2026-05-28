import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import pandas as pd

# ==========================================
# 1. CẤU HÌNH TRANG WEB & HEADER
# ==========================================
st.set_page_config(page_title="Mô phỏng lấy mẫu và khôi phục", page_icon="🎛️", layout="wide")

st.markdown("<h1 style='text-align: center; margin-bottom: 4px;'>🎛️ Mô phỏng lấy mẫu và khôi phục</h1>", unsafe_allow_html=True)
st.markdown("---")

# ==========================================
# 2. LÝ THUYẾT & CÔNG THỨC
# ==========================================
with st.expander("📖 Xem giải thích lý thuyết & Công thức toán học"):
    # Hàng 1: Nyquist | Sinc
    r1c1, r1c2 = st.columns(2)
    with r1c1:
        st.markdown("**1. Định lý lấy mẫu Nyquist-Shannon:**")
        st.markdown(
            "Để khôi phục tín hiệu gốc mà không bị chồng phổ (Aliasing), "
            "tần số lấy mẫu $f_s$ phải thỏa mãn:"
        )
        st.latex(r"f_s \ge 2 f_{max}")
    with r1c2:
        st.markdown("**2. Khôi phục lý tưởng — Nội suy Sinc:**")
        st.latex(
            r"x_r(t) = \sum_{n=-\infty}^{\infty} x(nT_s)"
            r"\cdot \mathrm{sinc}\!\left(\frac{t - nT_s}{T_s}\right)"
        )

    st.markdown("---")

    # Hàng 2: FFT | SNR  ← ngang hàng nhau
    r2c1, r2c2 = st.columns(2)
    with r2c1:
        st.markdown("**3. Phép biến đổi Fourier rời rạc (FFT):**")
        st.latex(r"X[k] = \sum_{n=0}^{N-1} x[n]\, e^{-j2\pi kn/N}")
        st.markdown("Biên độ được chuẩn hóa: $|X[k]| / N$")
    with r2c2:
        st.markdown("**4. Sai số tái tạo — SNR:**")
        st.latex(
            r"\mathrm{SNR} = 10\log_{10}"
            r"\!\left(\frac{\sum x(t)^2}{\sum [x(t)-x_r(t)]^2}\right) \text{ dB}"
        )

# ==========================================
# 3. HÀM TÍNH TOÁN LÕI
# ==========================================
MAX_SINC_SAMPLES = 400

@st.cache_data
def calculate_signals(f_sig, f_samp, wave_type):
    num_points = int(max(2000, 40 * max(f_sig, f_samp)))
    t_cont = np.linspace(0, 1, num_points)

    def make_wave(t, f, wtype):
        if wtype == 'Sin':
            return np.cos(2 * np.pi * f * t)
        elif wtype == 'Vuông':
            return signal.square(2 * np.pi * f * t)
        else:
            return signal.sawtooth(2 * np.pi * f * t, 0.5)

    x_cont = make_wave(t_cont, f_sig, wave_type)

    T_s = 1.0 / f_samp
    t_disc = np.arange(0, 1 + T_s, T_s)
    x_disc = make_wave(t_disc, f_sig, wave_type)

    # FIX SINC BOUNDARY: zero-pad ra ngoài [0,1] cả hai phía
    pad = min(int(f_samp), 50)
    t_disc_pad = np.arange(-pad * T_s, 1 + (pad + 1) * T_s, T_s)
    x_disc_pad = make_wave(t_disc_pad, f_sig, wave_type)

    # FIX PERFORMANCE: giới hạn số mẫu
    if len(t_disc_pad) > MAX_SINC_SAMPLES:
        idx = np.round(np.linspace(0, len(t_disc_pad) - 1, MAX_SINC_SAMPLES)).astype(int)
        t_disc_pad = t_disc_pad[idx]
        x_disc_pad = x_disc_pad[idx]

    # Vectorized Sinc Interpolation
    t_matrix = (t_cont[:, np.newaxis] - t_disc_pad[np.newaxis, :]) / T_s
    x_recon = x_disc_pad @ np.sinc(t_matrix).T

    # FFT
    N = len(x_disc)
    freqs_full = np.fft.fftfreq(N, T_s)
    fft_full   = np.abs(np.fft.fft(x_disc)) / N

    freqs_twosided = np.fft.fftshift(freqs_full)
    fft_twosided   = np.fft.fftshift(fft_full)

    pos_mask = freqs_full >= 0

    # Sai số & SNR
    error   = x_cont - x_recon
    mse     = np.mean(error ** 2)
    sig_pwr = np.mean(x_cont ** 2)
    snr_db  = 10 * np.log10(sig_pwr / (mse + 1e-12))

    return (
        t_cont, x_cont,
        t_disc, x_disc,
        x_recon, error, mse, snr_db,
        freqs_full[pos_mask], fft_full[pos_mask],
        freqs_twosided, fft_twosided,
    )

# ==========================================
# 4. BẢNG ĐIỀU KHIỂN (SIDEBAR)
# ==========================================
st.sidebar.markdown("### ⚙️ Bảng Điều Khiển")
wave_type = st.sidebar.radio("1. Chọn loại sóng", ("Sin", "Vuông", "Tam giác"))

f_sig  = st.sidebar.number_input(
    "2. Tần số tín hiệu (Hz)",
    min_value=1.0, max_value=500.0, value=2.0, step=0.5, format="%.1f"
)
f_samp = st.sidebar.number_input(
    "3. Tần số lấy mẫu (Hz)",
    min_value=2.0, max_value=1000.0, value=20.0, step=1.0, format="%.1f"
)

st.sidebar.markdown("---")
st.sidebar.markdown("**ℹ️ Giới hạn hiệu năng**")
st.sidebar.caption(
    f"Số mẫu Sinc tối đa: **{MAX_SINC_SAMPLES}**. "
    "Giúp tránh treo ứng dụng khi f_samp lớn."
)

# Tính toán
(
    t_c, x_c,
    t_d, x_d,
    x_r, error, mse, snr_db,
    freqs_pos, fft_pos,
    freqs_two, fft_two,
) = calculate_signals(f_sig, f_samp, wave_type)

# ==========================================
# 5. HỆ THỐNG CẢNH BÁO NYQUIST
# ==========================================
st.markdown("### 📊 Phân tích Thông số Lấy mẫu")
nyquist_req  = 2 * f_sig
nyquist_freq = f_samp / 2
aliasing     = f_samp < nyquist_req

col1, col2, col3, col4 = st.columns(4)
col1.metric("Tần số tín hiệu",     f"{f_sig} Hz")
col2.metric("Tần số lấy mẫu",      f"{f_samp} Hz")
col3.metric("Nyquist yêu cầu (≥)", f"{nyquist_req} Hz",
            delta=f"{f_samp - nyquist_req:+.1f} Hz",
            delta_color="normal" if not aliasing else "inverse")
col4.metric("SNR khôi phục",       f"{snr_db:.1f} dB",
            delta="Tốt" if snr_db > 20 else ("Trung bình" if snr_db > 5 else "Kém"),
            delta_color="normal" if snr_db > 20 else ("off" if snr_db > 5 else "inverse"))

if not aliasing:
    st.success(
        f"✅ **THỎA MÃN NYQUIST** — f_s ({f_samp} Hz) ≥ 2·f_sig ({nyquist_req} Hz). "
        f"Không xảy ra Aliasing. SNR = {snr_db:.1f} dB."
    )
elif f_samp > f_sig:
    st.warning(
        f"⚠️ **GẦN NGƯỠNG NYQUIST** — f_s ({f_samp} Hz) < 2·f_sig ({nyquist_req} Hz). "
        f"Có thể bị méo nhẹ. Cần tăng lên ít nhất {nyquist_req} Hz."
    )
else:
    st.error(
        f"🚨 **ALIASING NGHIÊM TRỌNG** — f_s ({f_samp} Hz) ≤ f_sig ({f_sig} Hz). "
        f"Tín hiệu khôi phục bị biến dạng hoàn toàn. SNR = {snr_db:.1f} dB."
    )

st.markdown("---")

# ==========================================
# 6. BIỂU ĐỒ THEO TAB
# ==========================================
COLOR_ORIG  = '#2E86AB'
COLOR_SAMP  = '#D63230'
COLOR_RECON = '#388659'
COLOR_ERR   = '#E07A5F'
COLOR_FFT   = '#8A4F7D'
COLOR_NYQ   = '#FF4444'

def style_ax(ax, title, title_color):
    ax.set_title(title, fontweight='bold', color=title_color, pad=8)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Gốc vs Khôi phục",
    "📍 Lấy mẫu",
    "📉 Sai số & SNR",
    "📊 Phổ một phía",
    "🔁 Phổ hai phía (Aliasing)",
])

# --- Tab 1: Overlay gốc + khôi phục ---
with tab1:
    st.markdown(
        "**So sánh trực tiếp tín hiệu gốc và tín hiệu khôi phục trên cùng một đồ thị.** "
        "Khi Aliasing xảy ra, hai đường sẽ lệch nhau rõ rệt."
    )
    fig1, ax1 = plt.subplots(figsize=(12, 4))
    ax1.plot(t_c, x_c, color=COLOR_ORIG,  linewidth=2.5, label='Tín hiệu gốc $x(t)$', zorder=3)
    ax1.plot(t_c, x_r, color=COLOR_RECON, linewidth=2,   label='Tín hiệu khôi phục $x_r(t)$',
             linestyle='--', zorder=2)
    ax1.scatter(t_d, x_d, color=COLOR_SAMP, s=40, zorder=4, label='Mẫu rời rạc $x(nT_s)$')
    ax1.set_xlim(0, 1); ax1.set_ylim(-1.6, 1.6)
    ax1.legend(loc='upper right', fontsize=9)
    style_ax(ax1, 'Tín hiệu gốc (xanh) vs Khôi phục (lá cây) — Aliasing làm hai đường lệch nhau', COLOR_ORIG)
    fig1.tight_layout()
    st.pyplot(fig1)

# --- Tab 2: Stem lấy mẫu ---
with tab2:
    st.markdown(
        "**Tín hiệu sau khi lấy mẫu.** "
        "Tần số lấy mẫu càng cao, các mẫu càng dày và khôi phục càng chính xác."
    )
    fig2, ax2 = plt.subplots(figsize=(12, 4))
    ax2.plot(t_c, x_c, color=COLOR_ORIG, linewidth=1.2, alpha=0.35, label='Tín hiệu gốc (tham chiếu)')
    ax2.stem(t_d, x_d, linefmt='r-', markerfmt='ro', basefmt='k-')
    ax2.set_xlim(0, 1); ax2.set_ylim(-1.6, 1.6)
    ax2.legend(fontsize=9)
    style_ax(ax2, f'Tín hiệu sau lấy mẫu $x(nT_s)$ — {len(t_d)} mẫu tại f_s = {f_samp} Hz', COLOR_SAMP)
    fig2.tight_layout()
    st.pyplot(fig2)

# --- Tab 3: Sai số & SNR — ngang hàng với FFT (2 cột) ---
with tab3:
    st.markdown(
        "**Sai số khôi phục** = hiệu giữa tín hiệu gốc và tín hiệu khôi phục. "
        "SNR càng cao (> 20 dB) thì khôi phục càng tốt."
    )

    col_err, col_fft = st.columns(2)

    # Cột trái: Sai số tái tạo
    with col_err:
        st.markdown("##### 📉 Sai số tái tạo")
        fig_err, (axE1, axE2) = plt.subplots(2, 1, figsize=(6, 5))

        axE1.plot(t_c, x_c,  color=COLOR_ORIG,  linewidth=2, label='Gốc $x(t)$')
        axE1.plot(t_c, x_r,  color=COLOR_RECON, linewidth=2, label='Khôi phục $x_r(t)$', linestyle='--')
        axE1.set_xlim(0, 1); axE1.set_ylim(-1.6, 1.6)
        axE1.legend(fontsize=8)
        style_ax(axE1, 'So sánh tín hiệu', COLOR_ORIG)

        axE2.fill_between(t_c, error, color=COLOR_ERR, alpha=0.6)
        axE2.axhline(0, color='black', linewidth=0.8)
        axE2.set_xlim(0, 1)
        style_ax(
            axE2,
            f'Sai số $e(t)$  |  MSE={mse:.4f}  |  SNR={snr_db:.1f} dB',
            COLOR_ERR,
        )
        fig_err.tight_layout()
        st.pyplot(fig_err)

        # Bảng chỉ số
        metrics_df = pd.DataFrame({
            'Chỉ số': ['MSE', 'SNR (dB)', 'Lỗi tối đa', 'Lỗi trung bình'],
            'Giá trị': [
                f"{mse:.6f}",
                f"{snr_db:.2f} dB",
                f"{np.max(np.abs(error)):.4f}",
                f"{np.mean(np.abs(error)):.4f}",
            ],
            'Nhận xét': [
                'Càng nhỏ càng tốt',
                '>20 dB: Tốt | 5–20: TB | <5: Kém',
                'Sai số cực đại',
                'Sai số trung bình',
            ]
        })
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    # Cột phải: Phổ FFT một phía (tham chiếu nhanh)
    with col_fft:
        st.markdown("##### 📊 Phổ tần số (FFT) — tham chiếu")
        fig_fft_ref, ax_fft_ref = plt.subplots(figsize=(6, 5))
        ax_fft_ref.stem(freqs_pos, fft_pos, linefmt='m-', markerfmt='mo', basefmt='k-')
        ax_fft_ref.axvline(
            x=nyquist_freq, color=COLOR_NYQ, linestyle='--', linewidth=2,
            label=f'Nyquist = {nyquist_freq:.1f} Hz'
        )
        ax_fft_ref.axvline(
            x=f_sig, color=COLOR_ORIG, linestyle=':', linewidth=1.5,
            label=f'f_sig = {f_sig} Hz'
        )
        ax_fft_ref.set_xlim(-0.5, max(f_samp * 0.6, f_sig * 3))
        ax_fft_ref.set_ylim(0, 1.2)
        ax_fft_ref.set_xlabel('Tần số (Hz)', fontstyle='italic')
        ax_fft_ref.set_ylabel('Biên độ', fontstyle='italic')
        ax_fft_ref.legend(fontsize=8)
        style_ax(ax_fft_ref, 'Phổ FFT | Đỏ đứt: Nyquist | Xanh chấm: f_sig', COLOR_FFT)
        fig_fft_ref.tight_layout()
        st.pyplot(fig_fft_ref)

# --- Tab 4: Phổ một phía + đường Nyquist ---
with tab4:
    st.markdown(
        "**Phổ biên độ một phía (Single-sided).** "
        "Đường đỏ đứt là tần số Nyquist $f_s/2$ — thành phần nào vượt qua đường này sẽ bị gập ngược lại gây Aliasing."
    )
    fig4, ax4 = plt.subplots(figsize=(12, 4))
    ax4.stem(freqs_pos, fft_pos, linefmt='m-', markerfmt='mo', basefmt='k-')
    ax4.axvline(x=nyquist_freq, color=COLOR_NYQ, linestyle='--', linewidth=2,
                label=f'Nyquist = {nyquist_freq:.1f} Hz')
    ax4.axvline(x=f_sig, color=COLOR_ORIG, linestyle=':', linewidth=1.5,
                label=f'f_sig = {f_sig} Hz')
    ax4.set_xlim(-0.5, max(f_samp * 0.6, f_sig * 3))
    ax4.set_ylim(0, 1.2)
    ax4.set_xlabel('Tần số (Hz)', fontstyle='italic')
    ax4.set_ylabel('Biên độ', fontstyle='italic')
    ax4.legend(fontsize=9)
    style_ax(ax4, 'Phổ tần số (FFT) — Một phía | Đường đỏ: Nyquist | Đường xanh: f_sig', COLOR_FFT)
    fig4.tight_layout()
    st.pyplot(fig4)

# --- Tab 5: Phổ hai phía ---
with tab5:
    st.markdown(
        "**Phổ biên độ hai phía (Two-sided / Double-sided).** "
        "Khi Aliasing xảy ra, các thành phần tần số cao bị **gập ngược** (fold) vào vùng $[-f_s/2, +f_s/2]$, "
        "chồng lên thành phần tín hiệu hợp lệ và gây méo."
    )
    fig5, ax5 = plt.subplots(figsize=(12, 4))
    ax5.stem(freqs_two, fft_two, linefmt='b-', markerfmt='bo', basefmt='k-')
    ax5.axvline( nyquist_freq, color=COLOR_NYQ, linestyle='--', linewidth=2,
                label=f'+Nyquist = +{nyquist_freq:.1f} Hz')
    ax5.axvline(-nyquist_freq, color=COLOR_NYQ, linestyle='--', linewidth=2,
                label=f'−Nyquist = −{nyquist_freq:.1f} Hz')
    ax5.axvspan(-nyquist_freq, nyquist_freq, alpha=0.06, color='green', label='Băng thông hợp lệ')
    ax5.set_xlim(-f_samp * 0.6, f_samp * 0.6)
    ax5.set_ylim(0, 1.2)
    ax5.set_xlabel('Tần số (Hz)', fontstyle='italic')
    ax5.set_ylabel('Biên độ', fontstyle='italic')
    ax5.legend(fontsize=9)
    style_ax(ax5, 'Phổ hai phía — Vùng xanh lá: băng thông hợp lệ | Đường đỏ: giới hạn Nyquist', COLOR_FFT)
    fig5.tight_layout()
    st.pyplot(fig5)

# ==========================================
# 7. XUẤT DỮ LIỆU
# ==========================================
st.markdown("---")
st.markdown("### 💾 Xuất Dữ liệu Phân tích")

col_dl1, col_dl2 = st.columns(2)

with col_dl1:
    df_fft = pd.DataFrame({'Tần số (Hz)': freqs_pos, 'Biên độ FFT': fft_pos})
    st.download_button(
        label="📥 Tải Phổ FFT (.CSV)",
        data=df_fft.to_csv(index=False).encode('utf-8'),
        file_name='pho_FFT.csv',
        mime='text/csv',
    )

with col_dl2:
    df_err = pd.DataFrame({
        'Thời gian (s)':       t_c,
        'Tín hiệu gốc':        x_c,
        'Tín hiệu khôi phục':  x_r,
        'Sai số e(t)':         error,
    })
    st.download_button(
        label="📥 Tải Sai số & Tín hiệu (.CSV)",
        data=df_err.to_csv(index=False).encode('utf-8'),
        file_name='sai_so_khoi_phuc.csv',
        mime='text/csv',
    )

st.caption(
    f"Thông số phiên này: wave={wave_type} | f_sig={f_sig} Hz | "
    f"f_samp={f_samp} Hz | N_mẫu={len(t_d)} | SNR={snr_db:.1f} dB"
)
