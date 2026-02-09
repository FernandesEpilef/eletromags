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
# =============================================

# passo_1: definição das cargas

# 1) Definição das cargas (Exercício 15.1)
# -----------------------------
charges = np.array([-1.0, +1.0, -1.0])  # valores de Qk, com Q/(4πϵ)=1 embutido
pos = np.array([[-1.0, 0.0],
                [ 0.0, 1.0],
                [ 1.0, 0.0]])           # posições (xk, yk)

# Janela pedida no PDF
XMIN, XMAX = -5.0, 5.0
YMIN, YMAX = -5.0, 5.0

# Passos Δℓ (o PDF sugere 0.1 ou 0.01)
DL_E = 0.10  # para linhas de campo (eqs. 15.5 e 15.6)
DL_V = 0.10  # para equipotenciais (eqs. 15.7 e 15.8)

# Quantidade de curvas e controles de laço
NLE = 16        # número de linhas de campo por carga (pontos iniciais no círculo)
NLV = 6         # número de equipotenciais por carga (pontos iniciais variando "raio")
MAX_STEPS = 5000
PLOT_EVERY = 2  # plota 1 ponto a cada PLOT_EVERY iterações (reduz densidade)

# Tolerâncias (para não explodir na singularidade e parar perto de cargas)
E_SING = 5e-5         # se |E| ficar muito pequeno, para
NEAR_CHARGE_E = 5e-2  # linhas de campo: para se chegar muito perto de uma carga
NEAR_CHARGE_V = 5e-3  # equipotenciais: ainda mais perto -> para


# -----------------------------
# 2) Funções do problema (eqs. 15.3 e 15.4)
# -----------------------------
def electric_field(x, y):
    """
    Campo elétrico E = (Ex,Ey) em (x,y).

    (PDF) eq. (15.3):
      Ex(x,y) = Σk [ Qk * (x - xk) / rk^3 ]
      Ey(x,y) = Σk [ Qk * (y - yk) / rk^3 ]
    onde rk = sqrt((x-xk)^2 + (y-yk)^2)
    e Q/(4πϵ)=1 já está embutido em Qk.
    """
    Ex = 0.0
    Ey = 0.0

    for k in range(len(charges)):
        dx = x - pos[k, 0]
        dy = y - pos[k, 1]
        r2 = dx*dx + dy*dy
        r = np.sqrt(r2)

        # evita divisão por zero (singularidade na carga)
        if r == 0:
            continue

        r3 = r**3
        Ex += charges[k] * dx / r3
        Ey += charges[k] * dy / r3

    return Ex, Ey


def potential_V(x, y):
    """
    Potencial elétrico V(x,y).

    (PDF) eq. (15.4):
      V(x,y) = Σk [ Qk / rk ]
    onde rk = sqrt((x-xk)^2 + (y-yk)^2)
    e Q/(4πϵ)=1 já está embutido.
    """
    V = 0.0

    for k in range(len(charges)):
        dx = x - pos[k, 0]
        dy = y - pos[k, 1]
        r = np.sqrt(dx*dx + dy*dy)

        if r == 0:
            continue

        V += charges[k] / r

    return V


# -----------------------------
# 3) Funções auxiliares simples
# -----------------------------
def inside_box(x, y):
    """Verifica se (x,y) está dentro da janela pedida no PDF."""
    return (XMIN <= x <= XMAX) and (YMIN <= y <= YMAX)


def near_any_charge(x, y, tol):
    """Para evitar singularidade: para se estiver muito perto de alguma carga."""
    for k in range(len(charges)):
        if abs(x - pos[k, 0]) < tol and abs(y - pos[k, 1]) < tol:
            return True
    return False


# -----------------------------
# 4) Traçar linhas de campo (eqs. 15.5 e 15.6)
# -----------------------------
def trace_field_line(x0, y0, q_source):
    """
    Traça uma linha de campo a partir do ponto inicial (x0,y0).

    (PDF) eqs. (15.5) e (15.6):
      Δx = Δℓ * Ex/|E|
      Δy = Δℓ * Ey/|E|
    com |E| = sqrt(Ex^2 + Ey^2)

    Observação comum (como no exemplo do livro): se a carga “fonte” for negativa,
    inverte-se o passo para desenhar linhas "saindo" dela (para visualização).
    """
    xs = [x0]
    ys = [y0]

    x = x0
    y = y0

    for step in range(MAX_STEPS):
        Ex, Ey = electric_field(x, y)
        E = np.sqrt(Ex*Ex + Ey*Ey)

        # (a) singularidade (quando |E| muito pequeno)
        if E <= E_SING:
            break

        # (PDF) eqs. (15.5)-(15.6): deslocamento normalizado por |E|
        dx = DL_E * (Ex / E)
        dy = DL_E * (Ey / E)

        # ajuste do exemplo: para carga negativa, inverte para "sair" dela
        if q_source < 0:
            dx = -dx
            dy = -dy

        x_new = x + dx
        y_new = y + dy

        # (c) se saiu da janela, para
        if not inside_box(x_new, y_new):
            break

        # (b) se chegou perto de uma carga, para (evitar r ~ 0)
        if near_any_charge(x_new, y_new, NEAR_CHARGE_E):
            break

        x = x_new
        y = y_new

        # salva ponto a cada PLOT_EVERY iterações (só para não ficar pesado)
        if step % PLOT_EVERY == 0:
            xs.append(x)
            ys.append(y)

    return np.array(xs), np.array(ys)


# -----------------------------
# 5) Traçar equipotenciais (eqs. 15.7 e 15.8)
# -----------------------------
def trace_equipotential(x0, y0, close_radius=0.2):
    """
    Traça uma equipotencial a partir do ponto inicial (x0,y0).

    Ideia do PDF: a equipotencial é sempre perpendicular a E.
    (PDF) eqs. (15.7) e (15.8):
      Δx = -Δℓ * Ey/|E|
      Δy =  Δℓ * Ex/|E|

    Condições de parada:
    - sair da janela
    - chegar perto de carga
    - fechar laço (voltar perto do ponto inicial)
    """
    xs = [x0]
    ys = [y0]

    x_start = x0
    y_start = y0

    # Tenta uma direção; se falhar por sair da janela, tenta a direção oposta 1 vez
    DIR = +1
    tried_both = False

    x = x0
    y = y0

    for step in range(MAX_STEPS):
        Ex, Ey = electric_field(x, y)
        E = np.sqrt(Ex*Ex + Ey*Ey)

        if E <= 5e-4:
            break

        # (PDF) eqs. (15.7)-(15.8): deslocamento perpendicular a E
        dx = -DL_V * (Ey / E)
        dy =  DL_V * (Ex / E)

        x_new = x + DIR * dx
        y_new = y + DIR * dy

        # se saiu da janela: tenta inverter direção uma vez
        if not inside_box(x_new, y_new):
            if not tried_both:
                tried_both = True
                DIR = -DIR
                x = x_start
                y = y_start
                xs = [x_start]
                ys = [y_start]
                continue
            else:
                break

        # se está perto de uma carga: para
        if near_any_charge(x_new, y_new, NEAR_CHARGE_V):
            break

        # checagem de fechamento do laço (voltar perto do início)
        dist0 = np.sqrt((x_new - x_start)**2 + (y_new - y_start)**2)
        if dist0 < close_radius and step > 50:
            break

        x = x_new
        y = y_new

        if step % PLOT_EVERY == 0:
            xs.append(x)
            ys.append(y)

    return np.array(xs), np.array(ys)


# -----------------------------
# 6) Pontos iniciais
# -----------------------------
def field_start_points(radius=0.10):
    """
    Pontos iniciais das linhas de campo:
    - distribui NLE pontos num círculo pequeno ao redor de cada carga (como no exemplo do PDF/livro).
      x0 = xk + r cos(theta)
      y0 = yk + r sin(theta)
    """
    starts = []
    for k in range(len(charges)):
        for i in range(NLE):
            theta = 2*np.pi * i / NLE
            x0 = pos[k, 0] + radius*np.cos(theta)
            y0 = pos[k, 1] + radius*np.sin(theta)
            starts.append((x0, y0, charges[k]))
    return starts


def equip_start_points(angle_deg=45.0, r0=0.5):
    """
    Pontos iniciais das equipotenciais:
    - escolhe um ângulo fixo e vai dobrando o raio (fator 2) para gerar várias curvas.
    """
    starts = []
    ang = np.deg2rad(angle_deg)

    for k in range(len(charges)):
        factor = r0
        for _ in range(NLV):
            x0 = pos[k, 0] + factor*np.cos(ang)
            y0 = pos[k, 1] + factor*np.sin(ang)
            if inside_box(x0, y0):
                starts.append((x0, y0))
            factor *= 2.0
    return starts


# -----------------------------
# 7) Programa principal: calcula e plota
# -----------------------------
def main():
    plt.figure(figsize=(7, 7))
    plt.title("Linhas de campo (azul) e equipotenciais (preto) — Exercício 15.1")
    plt.xlim(XMIN, XMAX)
    plt.ylim(YMIN, YMAX)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.grid(True, alpha=0.25)

    # Plota as cargas (visual)
    for q, (xq, yq) in zip(charges, pos):
        if q > 0:
            plt.scatter([xq], [yq], marker="+", s=150)
        else:
            plt.scatter([xq], [yq], marker="_", s=200)

    # (A) Linhas de campo (eqs. 15.5-15.6)
    for (x0, y0, qsrc) in field_start_points(radius=0.10):
        xs, ys = trace_field_line(x0, y0, qsrc)
        if len(xs) > 2:
            plt.plot(xs, ys, linewidth=1.2)

    # (B) Equipotenciais (eqs. 15.7-15.8)
    for (x0, y0) in equip_start_points(angle_deg=45.0, r0=0.5):
        xs, ys = trace_equipotential(x0, y0, close_radius=0.2)
        if len(xs) > 2:
            plt.plot(xs, ys, "k-", linewidth=1.0)

    plt.show()


if __name__ == "__main__":
    main()

