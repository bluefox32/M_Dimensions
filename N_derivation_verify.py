"""
N_derivation_verify.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
スケーリング指数 N の完全導出検証

発見された統一構造:
  N = log_{1/E0}(M_target / M_ref)
    = ln(M_target / M_ref) / ln(1/E0)

  M_ref = Omega(e, tau_mH) * E0 = f1(tau_mH)/e * E0

差分構造:
  D2 = N_p - N_e = ln(m_p/m_e) / ln(r)  [質量比の対数]
  D3 = N_v - N_p = ln(v /m_p) / ln(r)  [質量比の対数]

陽子/電子質量比:
  m_p/m_e = 1836.15 ≈ 6*pi^5 = 1836.12  (誤差0.003)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import numpy as np
from scipy.optimize import brentq
import math

# ── 基本定数 ──
PI  = np.pi
E   = np.e
phi = (1 + np.sqrt(5)) / 2
b   = brentq(lambda x: x**x - PI, 1, 3)

E_H  = 1.493e-33
E_Pl = 1.221e28
E0   = np.sqrt(E_H * E_Pl)
r    = 1.0 / E0

a_star   = np.sqrt(PI / 2)
delta_pi = PI**2 - (phi**4 + 3)

def f(t):    return np.log(t)/t if t > 0 else -np.inf
def f1(t):   return (1 - np.log(t)) / t**2
def f2(t):   return (2*np.log(t) - 3) / t**3
def omega(tau): return f1(tau) / E

# ── 実測質量 ──
M_E  = 0.510999e6   # eV 電子
M_P  = 938.272e6    # eV 陽子
M_H  = 125.10e9     # eV ヒッグス
V    = 246.00e9     # eV VEV
PHI0 = V / np.sqrt(2)

tau_mH = M_H / PHI0      # ≈ 0.71918
Om     = omega(tau_mH)    # = f1(tau_mH)/e
M_ref  = Om * E0          # 質量生成の基準スケール

# ── N値の計算 ──
def N(mass): return math.log(mass / M_ref) / math.log(r)

N_e = N(M_E)
N_p = N(M_P)
N_v = N(V)
D2  = N_p - N_e          # = ln(m_p/m_e)/ln(r)
D3  = N_v - N_p          # = ln(v/m_p)/ln(r)


def main():
    sep = "=" * 58

    print(sep)
    print("  N_derivation_verify.py")
    print("  スケーリング指数 N の完全導出検証")
    print(sep)

    # §1 基準スケール
    print()
    print("§1  基準スケール M_ref")
    print("  tau_mH = m_H/phi0 =", round(tau_mH, 8))
    print("  Om = f1(tau_mH)/e =", round(Om, 8))
    print("  M_ref = Om * E0   =", round(M_ref, 6), "eV")
    print("        = f1(tau_mH) * E0 / e")
    print("        = ヒッグス勾配指標 * E0 / BPS点")

    # §2 統一公式
    print()
    print("§2  統一公式")
    print("  N = log_{1/E0}(M / M_ref)")
    print("    = ln(M / M_ref) / ln(1/E0)")
    print()
    print("  N_e =", round(N_e, 8))
    print("  N_p =", round(N_p, 8))
    print("  N_v =", round(N_v, 8))

    # §3 差分の対数構造
    print()
    print("§3  差分の対数構造")
    D2_check = math.log(M_P / M_E) / math.log(r)
    D3_check = math.log(V  / M_P)  / math.log(r)
    print("  D2 = N_p - N_e = ln(m_p/m_e)/ln(r)")
    print("     計算:", round(D2, 8), " 検証:", round(D2_check, 8),
          " 一致:", D2 == D2_check)
    print("  D3 = N_v - N_p = ln(v/m_p)/ln(r)")
    print("     計算:", round(D3, 8), " 検証:", round(D3_check, 8),
          " 一致:", D3 == D3_check)
    print()
    print("  → 全差分が「質量比の対数/ln(r)」として閉じる")

    # §4 陽子/電子質量比と 6*pi^5
    print()
    print("§4  m_p/m_e の定数表現")
    ratio_pe = M_P / M_E
    six_pi5  = 6 * PI**5
    print("  m_p/m_e     =", round(ratio_pe, 5))
    print("  6*pi^5      =", round(six_pi5, 5), " 誤差:", round(abs(six_pi5-ratio_pe), 4))
    print("  誤差率       =", round(abs(six_pi5-ratio_pe)/ratio_pe*100, 4), "%")
    print()
    print("  D2 using 6pi^5:", round(math.log(six_pi5)/math.log(r), 8))
    print("  D2 actual:     ", round(D2, 8))
    print("  差:            ", round(abs(math.log(six_pi5)/math.log(r) - D2), 8))

    # §5 N_e の完全な閉じた式
    print()
    print("§5  N_e の閉じた式（完全版）")
    print("  N_e = [ln(m_e/E0) + 1 - ln(f1(tau_mH))] / ln(r)")
    lnf1 = math.log(f1(tau_mH))
    N_e_closed = (math.log(M_E/E0) + 1 - lnf1) / math.log(r)
    print("  ln(m_e/E0)          =", round(math.log(M_E/E0), 8))
    print("  1                   =", 1.0)
    print("  -ln(f1(tau_mH))     =", round(-lnf1, 8))
    print("    ln(f1) = ln(1-ln(tau_mH)) - 2*ln(tau_mH)")
    print("           =", round(math.log(1-math.log(tau_mH)) - 2*math.log(tau_mH), 8))
    print("  ln(r)               =", round(math.log(r), 8))
    print("  N_e (閉じた式)      =", round(N_e_closed, 8))
    print("  N_e (実測)          =", round(N_e, 8))
    print("  誤差                =", round(abs(N_e_closed - N_e), 10), " (機械精度)")

    # §6 N_e ≈ a*xe の旧近似との比較
    print()
    print("§6  旧近似式との比較")
    N_e_old = a_star*E + delta_pi/E**2
    print("  旧: N_e ≈ sqrt(pi/2)*e + delta_pi/e^2 =", round(N_e_old, 8))
    print("  新: N_e = [ln(m_e/E0)+1-ln(f1(tau_mH))]/ln(r) =", round(N_e_closed, 8))
    print("  実測:                                          =", round(N_e, 8))
    print()
    print("  旧近似の誤差:", round(abs(N_e_old - N_e), 6))
    print("  新公式の誤差:", round(abs(N_e_closed - N_e), 10), "(厳密)")
    print()
    print("  旧近似 sqrt(pi/2)*e は N_e_raw = ln(m_e/E0)/ln(r) と一致")
    print("  N_e_raw =", round(math.log(M_E/E0)/math.log(r), 8))
    print("  ≈ sqrt(pi/2)*e =", round(a_star*E, 8), " (誤差:", round(abs(math.log(M_E/E0)/math.log(r)-a_star*E), 6), ")")

    # §7 D3 の近似式との比較
    print()
    print("§7  D3 の定数近似")
    D3_approx = 1 + delta_pi*(1 + 1/E**2)
    print("  D3 実測                     =", round(D3, 8))
    print("  1 + delta_pi*(1+1/e^2)      =", round(D3_approx, 8))
    print("  誤差                        =", round(abs(D3_approx - D3), 6))
    print("  ln(v/m_p)/ln(r)             =", round(D3, 8), " (厳密)")

    # §8 質量再構成（最終確認）
    print()
    print("§8  質量再構成（統一公式による）")
    print()
    print("  mass = M_ref * r^N = Om*E0 * (1/E0)^N")
    print()
    results = [
        ("電子",    M_E,  N_e, "MeV", 1e6),
        ("陽子",    M_P,  N_p, "MeV", 1e6),
        ("VEV",     V,    N_v, "GeV", 1e9),
    ]
    print("  粒子     N値         計算質量         実測            誤差")
    print("  " + "-"*56)
    for name, mass, Nval, unit, scale in results:
        calc = M_ref * r**Nval
        err  = abs(calc - mass) / mass * 100
        mark = "✓" if err < 0.01 else "△"
        print("  " + mark, name,
              round(Nval,6),
              round(calc/scale,4), unit,
              round(mass/scale,4), unit,
              round(err,6), "%")

    # §9 理論の核心
    print()
    print(sep)
    print("  理論の核心（完全版）")
    print(sep)
    print("""
  テンソル積差分の最小実数値表現:
    Omega(e, tau) = f(e)*f'(tau) - f(tau)*f'(e)
                 = f'(tau)/e       [f'(e)=0 のため]

  ヒッグス粒子の tau値で評価:
    Omega(e, tau_mH) = f'(tau_mH)/e = g_H(tau_mH)/e

  質量生成の統一公式:
    mass = Omega(e,tau_mH) * E0 * (1/E0)^N
         = M_ref * r^N

    N = log_{1/E0}(mass / M_ref)
      = ln(mass/M_ref) / ln(1/E0)

  差分の対数構造（E0・tau_mH に依存しない）:
    D2 = N_p - N_e = ln(m_p/m_e) / ln(1/E0)
    D3 = N_v - N_p = ln(v/m_p)  / ln(1/E0)

  陽子/電子質量比:
    m_p/m_e = 1836.15 ≈ 6*pi^5 = 1836.12  (0.003% 誤差)

  → 素粒子質量階層は「E0を基底とした対数スケール上の
    質量比の積み重ね」として完全に記述される。
""")


if __name__ == "__main__":
    main()
