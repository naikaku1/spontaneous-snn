import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)

# ================= 基本設定 =================
N, N_EXC   = 800, 640      # 全ニューロン数 / うち興奮性
N_CLUSTER  = 4             # 興奮性を何群に分けるか
DT         = 1.0           # ms

# ---- 3要素のパラメータ ----
W_EXC   = 0.6              # 興奮性シナプス強度
W_INH   = 3.0             # 側方抑制の強度（群どうしを抑え合わせる）
GAIN    = 0.7             # シナプス入力の全体スケール
I_BIAS  = 0.06            # ★内因性電流（外部入力ゼロでも活動を保つ火種）
P_IN    = 0.4             # 群内の結合確率
ADAPT_STR = 0.6           # ★適応電流の強さ（発火すると疲れて発火しにくくなる）
ADAPT_TAU = 80.0          # ★適応の回復時定数(ms)


def _pink(n):
    """1/f（ピンク）ノイズ。脳の「バグ・退屈」に相当する外部刺激。"""
    white = rng.standard_normal(n)
    f = np.fft.rfftfreq(n); f[0] = f[1]
    return np.fft.irfft(np.fft.rfft(white) / np.sqrt(f), n).astype(np.float32)


def make_W(seed=0):
    """クラスタ構造つき結合行列。
    興奮性は自群内で密結合、抑制性は他群を抑える（側方抑制）。"""
    r = np.random.default_rng(seed)
    W = np.zeros((N, N), dtype=np.float32)
    csize    = N_EXC // N_CLUSTER
    inh_size = (N - N_EXC) // N_CLUSTER
    # 興奮性ニューロン：自群の興奮＋自群の抑制ニューロンを駆動
    for i in range(N_EXC):
        ci = i // csize
        lo, hi = ci * csize, (ci + 1) * csize
        m = (r.random(hi - lo) < P_IN).astype(np.float32)
        W[i, lo:hi] = m * r.random(hi - lo).astype(np.float32) * W_EXC
        ilo, ihi = N_EXC + ci * inh_size, N_EXC + (ci + 1) * inh_size
        m2 = (r.random(ihi - ilo) < 0.3).astype(np.float32)
        W[i, ilo:ihi] = m2 * r.random(ihi - ilo).astype(np.float32) * W_EXC
    # 抑制性ニューロン：自群以外の興奮性ニューロンを抑える
    for i in range(N_EXC, N):
        ci = (i - N_EXC) // inh_size
        for tc in range(N_CLUSTER):
            if tc == ci:
                continue
            lo, hi = tc * csize, (tc + 1) * csize
            m = (r.random(hi - lo) < 0.3).astype(np.float32)
            W[i, lo:hi] = -m * r.random(hi - lo).astype(np.float32) * W_INH
    np.fill_diagonal(W, 0.0)
    return W, csize


def simulate(steps, input_on_steps, seed=0):
    """LIF ニューロン + 不応期 + 内因性電流 + スパイク適応。
    input_on_steps を境に外部ノイズを OFF にする。"""
    W, csize = make_W(seed)
    v = (rng.random(N) * 0.5).astype(np.float32)
    tau, thr0, REFRAC = 20.0, 1.0, 5
    refrac = np.zeros(N, dtype=np.int32)
    adapt  = np.zeros(N, dtype=np.float32)          # 疲労変数
    spikes = np.zeros((steps, N), dtype=bool)

    r2 = np.random.default_rng(seed + 1)
    bias = np.zeros(N, dtype=np.float32)
    bias[:N_EXC] = I_BIAS * (0.5 + r2.random(N_EXC))  # 内因性電流（個体差つき）

    inj = np.arange(0, N, 50)
    noise = np.zeros((steps, N), dtype=np.float32)
    noise[:, inj] = np.stack([_pink(steps) for _ in inj]).T * 2.0

    prev = np.zeros(N, dtype=np.float32)
    for t in range(steps):
        I = (W.T @ prev) * GAIN + bias
        if t < input_on_steps:
            I += noise[t]
        active = refrac <= 0
        v = np.where(active, v + (-v / tau + I) * DT, 0.0).astype(np.float32)
        thr = thr0 + adapt                          # 疲れたぶん発火しにくい
        fired = (v >= thr) & active
        v[fired] = 0.0
        refrac[fired] = REFRAC
        refrac -= 1
        adapt = adapt * (1 - DT / ADAPT_TAU) + fired.astype(np.float32) * ADAPT_STR
        spikes[t] = fired
        prev = fired.astype(np.float32)
    return spikes, csize


def main():
    STEPS, OFF = 3500, 1200
    sp, csize = simulate(STEPS, OFF, seed=0)
    tail = sp[-400:].mean() * 1000
    print(f"入力OFF後の最終盤発火率: {tail:.1f} Hz  → 永続自走" if tail > 1
          else f"最終盤 {tail:.1f} Hz → 沈黙")

    colors = ['#e74c3c', '#2980b9', '#27ae60', '#8e44ad']
    fig, ax = plt.subplots(2, 1, figsize=(16, 10),
                           gridspec_kw={'height_ratios': [3, 1.3]})
    for c in range(N_CLUSTER):
        lo, hi = c * csize, (c + 1) * csize
        tt, nn = np.where(sp[:, lo:hi])
        ax[0].scatter(tt, nn + lo, s=0.4, c=colors[c], label=f'group {c}')
    tt, nn = np.where(sp[:, N_EXC:])
    ax[0].scatter(tt, nn + N_EXC, s=0.3, c='gray', alpha=0.5)
    ax[0].axvline(OFF, color='k', ls='--', lw=2)
    ax[0].text(OFF + 20, N - 30, 'input OFF', fontsize=11)
    ax[0].set_ylabel('neuron ID'); ax[0].legend(loc='upper right', markerscale=15)
    ax[0].set_title('Spontaneous activity: groups self-organize into call-and-response after input OFF')
    for c in range(N_CLUSTER):
        lo, hi = c * csize, (c + 1) * csize
        gr = np.convolve(sp[:, lo:hi].mean(axis=1), np.ones(40) / 40, 'same') * 1000
        ax[1].plot(gr, c=colors[c], lw=1.0, label=f'group {c}')
    ax[1].axvline(OFF, color='k', ls='--', lw=2)
    ax[1].set_xlabel('time (ms)'); ax[1].set_ylabel('group rate (Hz)')
    ax[1].legend(loc='upper right')
    plt.tight_layout(); plt.savefig('spontaneous_result.png', dpi=130)
    print("saved spontaneous_result.png")


if __name__ == "__main__":
    main()
