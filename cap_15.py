import numpy as np
import matplotlib.pyplot as plt

# =============================================
# Aluno: Felipe Fernandes
# Atividade: Capítulo 15 - Exercício 1

# Calcular E(x,y) e V(x,y) para cargas pontuais (eqs. 15.3 e 15.4)
# Traçar linhas de campo (eqs. 15.5 e 15.6)s
# Traçar equipotenciais (eqs. 15.7 e 15.8)
# Exercício 15.1: 3 cargas: -Q em (-1,0), +Q em (0,1), -Q em (1,0)
# Considere Q/(4*pi*epsilon) = 1
# Domínio: -5 <= x,y <= 5
# Step: 0.1 or 0.01

# ============================================================
#  PLOTAGEM DE CAMPO (Seção 15.2)
#  - Linhas de campo elétrico e linhas equipotenciais
#  - Cargas pontuais coplanares (2D)
#
#  Convenção do enunciado:
#  Considere Q/(4*pi*epsilon) = 1  (constante já "embutida")
#  e o domínio -5 <= x,y <= 5
# ============================================================


# -------------------------------
# Equações do livro (15.3) e (15.4)
# -------------------------------
def electric_field(x, y, Q, pos):
    """
    Campo elétrico E = (Ex, Ey) devido a cargas pontuais coplanares.
    Implementa a forma 2D da eq. (15.3) do PDF:
        Ex = sum_k Qk * (x - xk) / R^3
        Ey = sum_k Qk * (y - yk) / R^3
    onde R = sqrt((x-xk)^2 + (y-yk)^2)
    """
    Ex, Ey = 0.0, 0.0
    for qk, (xk, yk) in zip(Q, pos):
        dx = x - xk
        dy = y - yk
        R = np.sqrt(dx*dx + dy*dy)
        # Para evitar divisão por zero, o chamador também checa distância mínima
        Ex += qk * dx / (R**3)
        Ey += qk * dy / (R**3)
    return Ex, Ey


def potential(x, y, Q, pos):
    """
    Potencial elétrico V devido a cargas pontuais coplanares.
    Implementa a forma 2D da eq. (15.4) do PDF:
        V = sum_k Qk / R
    """
    V = 0.0
    for qk, (xk, yk) in zip(Q, pos):
        dx = x - xk
        dy = y - yk
        R = np.sqrt(dx*dx + dy*dy)
        V += qk / R
    return V


# -------------------------------
# Traçado numérico de uma linha de campo
# (passos 1-4 do livro para linhas de campo)
# usando eqs. (15.5) e (15.6):
#    dx = Δℓ * Ex/E
#    dy = Δℓ * Ey/E
# -------------------------------
def trace_field_line(
    start_xy, Q, pos,
    dL=0.05,                 # Δℓ (passo ao longo da linha de campo) ## IMPORTANTE E MEXE MUITO COM O RESULTADO
    bounds=5.0,              # janela: -bounds <= x,y <= bounds
    min_dist_charge=0.05,    # para evitar singularidade perto da carga
    Emin=5e-5,               # "E ~ 0" (ponto singular, como no exemplo)
    max_steps=5000,
    direction=+1             # +1 segue E; -1 segue -E (útil para completar a linha)
):
    x, y = start_xy
    pts = [(x, y)]

    for _ in range(max_steps):
        # Checa se saiu do domínio
        if abs(x) > bounds or abs(y) > bounds:
            break

        # Checa proximidade de qualquer carga (evitar singularidade)
        for (xk, yk) in pos:
            if np.hypot(x - xk, y - yk) < min_dist_charge:
                return np.array(pts)

        Ex, Ey = electric_field(x, y, Q, pos)
        E = np.hypot(Ex, Ey)

        # Checa ponto singular (E muito pequeno)
        if E < Emin:
            break

        # Eqs. (15.5) e (15.6) do PDF
        dx = direction * dL * (Ex / E)
        dy = direction * dL * (Ey / E)

        x += dx
        y += dy
        pts.append((x, y))

    return np.array(pts)


# -------------------------------
# Traçado numérico de uma linha equipotencial
# (passos 1-4 do livro para equipotenciais)
# usando eqs. (15.7) e (15.8), isto é, passo perpendicular a E:
#    dx = -Δℓ * Ey/E
#    dy =  Δℓ * Ex/E
# -------------------------------
def trace_equipotential(
    start_xy, Q, pos,
    dL=0.05,                 # Δℓ (passo ao longo da equipotencial)
    bounds=5.0,
    min_dist_charge=0.05,
    Emin=5e-5,
    max_steps=8000,
    close_tol=0.15,          # tolerância para "fechar" o laço
    min_steps_before_close=50,
    direction=+1
):
    xs, ys = start_xy
    x, y = xs, ys
    pts = [(x, y)]

    for n in range(max_steps):
        if abs(x) > bounds or abs(y) > bounds:
            break

        for (xk, yk) in pos:
            if np.hypot(x - xk, y - yk) < min_dist_charge:
                break
        else:
            Ex, Ey = electric_field(x, y, Q, pos)
            E = np.hypot(Ex, Ey)
            if E < Emin:
                break

            # Eqs. (15.7) e (15.8): direção perpendicular a E
            dx = direction * (-dL * (Ey / E))
            dy = direction * ( dL * (Ex / E))

            x += dx
            y += dy
            pts.append((x, y))

            # Checagem de fechamento (como no exemplo do PDF)
            if n > min_steps_before_close:
                if np.hypot(x - xs, y - ys) < close_tol:
                    break

            continue

        # se caiu no "break" de proximidade de carga
        break

    return np.array(pts)


# -------------------------------
# Função principal: plota linhas
# -------------------------------
def plot_field_and_equipotentials(
    Q, pos,
    bounds=5.0,
    # parâmetros das linhas de campo PODEM SER ALTERADOS
    NLE=12, r_start=0.15, dL_E=0.05,
    # parâmetros das equipotenciais PODEM SER ALTERADOS
    radii_V=(0.6, 1.0, 1.6, 2.2), angle_V=np.deg2rad(45), dL_V=0.05,
):
    pos = np.array(pos, dtype=float)
    Q = np.array(Q, dtype=float)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-bounds, bounds)
    ax.set_ylim(-bounds, bounds)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Linhas de campo elétrico (contínuas) e equipotenciais (tracejadas)")

    # Plota as cargas
    for qk, (xk, yk) in zip(Q, pos):
        if qk > 0:
            ax.plot(xk, yk, "o", markersize=10)
            ax.text(xk + 0.1, yk + 0.1, f"+{qk:g}")
        else:
            ax.plot(xk, yk, "s", markersize=8)
            ax.text(xk + 0.1, yk + 0.1, f"{qk:g}")

    # -------- Linhas de campo E --------
    # Pontos de partida em um "pequeno círculo" em torno de cada carga (como sugerido no exemplo)
    # x_s = x_Q + r cos(theta), y_s = y_Q + r sin(theta)
    for (qk, (xk, yk)) in zip(Q, pos):
        for i in range(NLE):
            theta = 2*np.pi * i / NLE
            xs = xk + r_start*np.cos(theta)
            ys = yk + r_start*np.sin(theta)

            # Para desenhar linhas que "vão do + para o -", fazemos:
            # - se a carga de partida é positiva, seguimos +E
            # - se a carga de partida é negativa, seguimos -E (isso ajuda a "emanar" da carga também)
            direction = +1 if qk > 0 else -1

            line = trace_field_line(
                (xs, ys), Q, pos,
                dL=dL_E, bounds=bounds, direction=direction
            )
            if len(line) > 2:
                ax.plot(line[:, 0], line[:, 1], linewidth=1.0)

    # -------- Equipotenciais --------
    # Pontos de partida escolhidos a partir de cada carga, com raio variando (radii_V)
    # e ângulo fixo (por ex. 45°), como discutido no texto do exemplo.
    for (xk, yk) in pos:
        for r in radii_V:
            xs = xk + r*np.cos(angle_V)
            ys = yk + r*np.sin(angle_V)

            # Traça nos dois sentidos para aumentar chance de fechar a curva
            for direction in (+1, -1):
                eq = trace_equipotential(
                    (xs, ys), Q, pos,
                    dL=dL_V, bounds=bounds, direction=direction
                )
                if len(eq) > 10:
                    ax.plot(eq[:, 0], eq[:, 1], linestyle="--", linewidth=1.0)

    ax.grid(True, alpha=0.3)
    plt.show()


# ============================================================
# EXEMPLO PRÁTICO (Exercício 15.1 do PDF):
# Três cargas: -Q, +Q, -Q em (-1,0), (0,1), (1,0)
# Com Q/(4*pi*epsilon) = 1
# ============================================================

if __name__ == "__main__":
    Q = [-1, +1, -1]
    pos = [(-1, 0), (0, 1), (1, 0)]
    plot_field_and_equipotentials(Q, pos)
