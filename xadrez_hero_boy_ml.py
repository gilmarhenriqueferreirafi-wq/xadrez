"""
XADREZ: HERO x BOY  (versao com MACHINE LEARNING)
=================================================
Computacao cognitiva: ninguem ensina estrategia de xadrez para eles.
Hero e Boy jogam, olham o resultado (vitoria / empate / derrota) e AJUSTAM o
proprio modelo para reconhecer padroes que levam a vencer. Aprendem o tempo todo.

  HERO = REDE NEURAL (768 entradas = quais pecas estao em quais casas -> 32 neuronios -> nota da posicao)
         Ela descobre sozinha padroes e conexoes entre pecas e casas.
  BOY  = REGRESSAO com 14 padroes (material, mobilidade, centro, roque, desenvolvimento...)
         Ele aprende QUANTO cada padrao vale (da pra ver os pesos na aba Boy_Modelo).

Como aprendem (aprendizado por reforco com replay):
  1. Cada lance gera uma "experiencia": posicao resultante + resultado final da partida
     (valor menor para lances mais distantes do fim: gamma = 0.97).
  2. As experiencias vao para um DATASET COMPARTILHADO (os dois aprendem com as partidas um do outro).
  3. Depois de cada partida cada modelo treina varias vezes em lotes sorteados do dataset
     (backpropagation + otimizador Adam) e o erro de previsao e registrado.
  4. Exploracao: no comeco arriscam lances novos; com o tempo exploram menos (usam o que aprenderam).
  5. Tabela compartilhada: vitorias/empates/derrotas por lance, consultada pelos dois.

Tudo fica salvo no Excel  memoria_xadrez_ml.xlsx  (mesma pasta do script):
  Placar | Partidas (com o erro de cada modelo) | Hero_Rede + Hero_Resto (pesos da rede)
  Boy_Modelo (pesos dos padroes) | Experiencias (dataset) | Compartilhada (lances)

Hero e Boy alternam quem joga de brancas (quem comeca) a cada partida, inclusive entre execucoes.

CONTROLES (trocam a qualquer momento):
  A = automatico (mostra os lances com pausa)      R = rapido (so o resultado de cada partida)
  L = ESPACO avanca um lance                       P = ESPACO avanca uma partida inteira
  ESPACO (no automatico) = pausa e entra no modo L Q = salva e sai

Instalar:  pip install chess openpyxl numpy
Rodar:     python xadrez_hero_boy_ml.py      (CMD, PowerShell ou Terminal)
IMPORTANTE: feche o Excel antes de rodar, senao o arquivo nao consegue ser salvo.
"""
import os
import random
import sys
import time
from datetime import datetime

import chess
import numpy as np
from openpyxl import Workbook, load_workbook
from openpyxl.chart import LineChart, Reference
from openpyxl.styles import Font
from openpyxl.utils import get_column_letter

PASTA = os.path.dirname(os.path.abspath(__file__))
ARQUIVO = os.path.join(PASTA, "memoria_xadrez_ml.xlsx")

MAX_LANCES = 160          # meio-lances; passou disso a partida e encerrada
PAUSA_AUTO = 0.4          # segundos entre lances no modo automatico
PESO_COMPARTILHADA = 0.3  # influencia da tabela compartilhada na escolha do lance

# --- Machine Learning ---
N_ENTRADA = 769           # 12 tipos de peca x 64 casas + material
OCULTA = 32               # neuronios do Hero
N_PADROES = 14            # padroes do Boy
GAMMA = 0.97              # desconto: lances proximos do fim valem mais
BUFFER_MAX = 3000         # tamanho do dataset (experiencias mais recentes)
LOTE = 64                 # tamanho do lote de treino
PASSOS_TREINO = 40        # lotes por partida, por modelo
MIN_TREINO = 64           # so treina quando houver experiencias suficientes
LR_HERO = 0.004
LR_BOY = 0.01

VALORES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
           chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
CENTRO = (chess.D4, chess.E4, chess.D5, chess.E5)

MOTIVOS = {
    "CHECKMATE": "xeque-mate",
    "STALEMATE": "afogamento",
    "INSUFFICIENT_MATERIAL": "material insuficiente",
    "SEVENTYFIVE_MOVES": "regra dos 75 lances",
    "FIFTY_MOVES": "regra dos 50 lances",
    "FIVEFOLD_REPETITION": "repeticao (5x)",
    "THREEFOLD_REPETITION": "repeticao (3x)",
    "LIMITE_MATERIAL": "limite de lances (vantagem de material)",
    "LIMITE_EMPATE": "limite de lances (igualdade)",
}
NOMES_MODO = {
    "auto": "AUTOMATICO",
    "rapido": "RAPIDO (sem tabuleiro)",
    "lance": "MANUAL - ESPACO a cada lance",
    "partida": "MANUAL - ESPACO a cada partida",
}
TECLAS_MODO = {"a": "auto", "r": "rapido", "l": "lance", "p": "partida"}

NOMES_PADROES = [
    "diferenca de peoes", "diferenca de cavalos", "diferenca de bispos",
    "diferenca de torres", "diferenca de damas", "par de bispos",
    "mobilidade do adversario", "minha mobilidade", "controle do centro",
    "avanco dos peoes", "xeque", "rei roqueado", "desenvolvimento das pecas",
    "vies (constante)",
]
NOMES_PECAS = ["peao", "cavalo", "bispo", "torre", "dama", "rei"]


# ---------------------------------------------------------------------------
# Tabuleiro -> numeros (o que os modelos "enxergam")
# ---------------------------------------------------------------------------
def material(board, cor):
    """Saldo de material do ponto de vista de 'cor'."""
    total = 0
    for peca, valor in VALORES.items():
        total += valor * len(board.pieces(peca, cor))
        total -= valor * len(board.pieces(peca, not cor))
    return total


def feats_rede(board, cor):
    """Entrada da rede neural: 768 casas (minhas pecas / pecas do adversario) + material.
    O tabuleiro e espelhado para as pretas, entao o modelo sempre 've' do ponto de vista de quem joga."""
    x = np.zeros(N_ENTRADA, dtype=np.float32)
    for casa, p in board.piece_map().items():
        if cor == chess.BLACK:
            casa = chess.square_mirror(casa)
        grupo = (0 if p.color == cor else 6) + p.piece_type - 1
        x[grupo * 64 + casa] = 1.0
    x[768] = material(board, cor) / 10.0
    return x


def feats_padroes(board, cor):
    """Os 14 padroes que o Boy avalia (todos do ponto de vista de 'cor')."""
    meu, adv = cor, not cor
    f = np.zeros(N_PADROES, dtype=np.float32)
    escalas = (4, 2, 2, 2, 1)
    tipos = (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)
    for i, t in enumerate(tipos):
        f[i] = (len(board.pieces(t, meu)) - len(board.pieces(t, adv))) / escalas[i]
    f[5] = (len(board.pieces(chess.BISHOP, meu)) >= 2) - (len(board.pieces(chess.BISHOP, adv)) >= 2)

    # mobilidade (lances possiveis) de quem esta na vez e de quem nao esta
    vez = board.pseudo_legal_moves.count() / 40
    board.push(chess.Move.null())
    fora = board.pseudo_legal_moves.count() / 40
    board.pop()
    f[6], f[7] = (vez, fora) if board.turn == adv else (fora, vez)

    centro = 0
    for casa in CENTRO:
        p = board.piece_at(casa)
        if p:
            centro += 1 if p.color == meu else -1
    f[8] = centro / 4

    def avanco(c):
        total = 0
        for casa in board.pieces(chess.PAWN, c):
            r = chess.square_rank(casa)
            total += r if c == chess.WHITE else 7 - r
        return total
    f[9] = (avanco(meu) - avanco(adv)) / 16

    if board.is_check():
        f[10] = 1.0 if board.turn == adv else -1.0

    def roque(c):
        rei = board.king(c)
        if rei is None:
            return 0
        fundo = 0 if c == chess.WHITE else 7
        return int(chess.square_rank(rei) == fundo and chess.square_file(rei) in (2, 6))
    f[11] = roque(meu) - roque(adv)

    def desenvolvidas(c):
        fundo = 0 if c == chess.WHITE else 7
        n = 0
        for t in (chess.KNIGHT, chess.BISHOP):
            for casa in board.pieces(t, c):
                if chess.square_rank(casa) != fundo:
                    n += 1
        return n
    f[12] = (desenvolvidas(meu) - desenvolvidas(adv)) / 4
    f[13] = 1.0
    return f


def desenhar(board):
    """Tabuleiro em texto: MAIUSCULAS = brancas, minusculas = pretas."""
    linhas = []
    for r in range(7, -1, -1):
        linha = f"{r + 1}  "
        for f in range(8):
            p = board.piece_at(chess.square(f, r))
            linha += (p.symbol() if p else ".") + " "
        linhas.append(linha)
    linhas.append("   a b c d e f g h")
    return "\n".join(linhas)


# ---------------------------------------------------------------------------
# Modelos de Machine Learning (so numpy)
# ---------------------------------------------------------------------------
class Adam:
    """Otimizador: ajusta os pesos na direcao que reduz o erro."""

    def __init__(self, params, lr):
        self.params = params
        self.lr = lr
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def passo(self, grads):
        self.t += 1
        for p, g, m, v in zip(self.params, grads, self.m, self.v):
            m *= 0.9
            m += 0.1 * g
            v *= 0.999
            v += 0.001 * g * g
            mh = m / (1 - 0.9 ** self.t)
            vh = v / (1 - 0.999 ** self.t)
            p -= self.lr * mh / (np.sqrt(vh) + 1e-8)


class RedeNeural:
    """769 entradas -> 32 neuronios (tanh) -> 1 saida (tanh) = nota da posicao entre -1 e +1."""

    def __init__(self):
        rng = np.random.default_rng()
        self.W1 = rng.normal(0, 0.05, (N_ENTRADA, OCULTA))
        self.W1[-1, :] = np.abs(rng.normal(1.5, 0.3, OCULTA))  # ponto de partida: material importa
        self.b1 = np.zeros(OCULTA)
        self.W2 = rng.normal(0.05, 0.05, (OCULTA, 1))
        self.b2 = np.zeros(1)
        self.otimizador = Adam([self.W1, self.b1, self.W2, self.b2], LR_HERO)

    def prever(self, X):
        h = np.tanh(X @ self.W1 + self.b1)
        return np.tanh(h @ self.W2 + self.b2)[:, 0]

    def treinar(self, X, y):
        n = len(X)
        h = np.tanh(X @ self.W1 + self.b1)
        saida = np.tanh(h @ self.W2 + self.b2)[:, 0]
        erro = saida - y
        dz2 = (2 * erro / n * (1 - saida ** 2))[:, None]
        dW2 = h.T @ dz2
        db2 = dz2.sum(axis=0)
        dz1 = (dz2 @ self.W2.T) * (1 - h ** 2)
        dW1 = X.T @ dz1
        db1 = dz1.sum(axis=0)
        self.otimizador.passo([dW1, db1, dW2, db2])
        return float(np.mean(erro ** 2))


class Regressao:
    """14 padroes x pesos -> tanh -> nota da posicao entre -1 e +1."""

    def __init__(self):
        self.w = np.zeros(N_PADROES)
        # ponto de partida: valor basico das pecas (o resto ele descobre sozinho)
        self.w[:5] = [0.4, 0.6, 0.6, 1.0, 0.9]
        self.otimizador = Adam([self.w], LR_BOY)

    def prever(self, F):
        return np.tanh(F @ self.w)

    def treinar(self, F, y):
        n = len(F)
        saida = np.tanh(F @ self.w)
        erro = saida - y
        dz = 2 * erro / n * (1 - saida ** 2)
        self.otimizador.passo([F.T @ dz])
        return float(np.mean(erro ** 2))


# ---------------------------------------------------------------------------
# Dataset compartilhado de experiencias e tabela compartilhada de lances
# ---------------------------------------------------------------------------
class Experiencias:
    """posicao (apos o lance) + de quem foi o lance + resultado esperado (-1..+1)."""

    def __init__(self):
        self.epd, self.cor, self.alvo, self.jogo = [], [], [], []
        self.X, self.F = [], []

    def __len__(self):
        return len(self.alvo)

    def adicionar(self, epd, cor, alvo, x, f, jogo):
        self.epd.append(epd)
        self.cor.append(cor)
        self.alvo.append(alvo)
        self.jogo.append(jogo)
        self.X.append(x)
        self.F.append(f)
        excesso = len(self.alvo) - BUFFER_MAX
        if excesso > 0:
            for lista in (self.epd, self.cor, self.alvo, self.jogo, self.X, self.F):
                del lista[:excesso]

    def lote(self, n):
        idx = random.sample(range(len(self.alvo)), min(n, len(self.alvo)))
        X = np.array([self.X[i] for i in idx])
        F = np.array([self.F[i] for i in idx])
        y = np.array([self.alvo[i] for i in idx])
        return X, F, y

    def carregar(self, linhas):
        for epd, cor, alvo, jogo in linhas:
            try:
                board = chess.Board()
                board.set_epd(epd)
            except ValueError:
                continue
            c = chess.WHITE if cor == "B" else chess.BLACK
            self.adicionar(epd, c, float(alvo), feats_rede(board, c), feats_padroes(board, c),
                           int(jogo or 0))

    def linhas(self):
        for epd, cor, alvo, jogo in zip(self.epd, self.cor, self.alvo, self.jogo):
            yield (epd, "B" if cor == chess.WHITE else "P", round(alvo, 4), jogo)


class Compartilhada:
    """{ posicao: { lance: [vitorias, empates, derrotas] } } do ponto de vista de quem fez o lance."""

    def __init__(self):
        self.dados = {}

    def pontuacao(self, epd, uci):
        r = self.dados.get(epd, {}).get(uci)
        if not r:
            return 0.0
        n = sum(r)
        return ((r[0] - r[2]) / n) * (min(n, 5) / 5)

    def registrar(self, epd, uci, idx):
        """idx: 0 = vitoria, 1 = empate, 2 = derrota."""
        self.dados.setdefault(epd, {}).setdefault(uci, [0, 0, 0])[idx] += 1

    def carregar(self, linhas):
        for epd, uci, v, e, d in linhas:
            self.dados.setdefault(epd, {})[uci] = [int(v or 0), int(e or 0), int(d or 0)]

    def linhas(self):
        for epd, lances in self.dados.items():
            for uci, (v, e, d) in lances.items():
                yield (epd, uci, v, e, d)


# ---------------------------------------------------------------------------
# Jogadores
# ---------------------------------------------------------------------------
class Agente:
    def __init__(self, nome, compartilhada):
        self.nome = nome
        self.comp = compartilhada
        self.historico = []   # (antes, lance_uci, depois, cor, x, f)
        self.ultimo_erro = None

    def novo_jogo(self):
        self.historico = []

    def jogar(self, board):
        antes = board.epd()
        cor = board.turn
        lance = self.escolher(board, antes, cor)
        board.push(lance)
        depois = board.epd()
        x = feats_rede(board, cor)
        f = feats_padroes(board, cor)
        board.pop()
        self.historico.append((antes, lance.uci(), depois, cor, x, f))
        return lance

    def _avaliar_candidatos(self, board, antes, cor, construir):
        """Retorna (lances, matriz_de_features, lance_que_da_mate_ou_None)."""
        lances = list(board.legal_moves)
        linhas = []
        for m in lances:
            board.push(m)
            if board.is_checkmate():
                board.pop()
                return lances, None, m
            linhas.append(construir(board, cor))
            board.pop()
        return lances, np.array(linhas), None

    def bonus_compartilhada(self, antes, lances):
        return PESO_COMPARTILHADA * np.array([self.comp.pontuacao(antes, m.uci()) for m in lances])


class Hero(Agente):
    """Rede neural. Explora com probabilidade epsilon (diminui com o tempo)."""

    def __init__(self, compartilhada):
        super().__init__("Hero", compartilhada)
        self.rede = RedeNeural()
        self.epsilon = 0.2

    def escolher(self, board, antes, cor):
        lances, X, mate = self._avaliar_candidatos(board, antes, cor, feats_rede)
        if mate:
            return mate
        if random.random() < self.epsilon:
            return random.choice(lances)
        valores = self.rede.prever(X) + self.bonus_compartilhada(antes, lances)
        valores = valores + np.random.uniform(-1e-3, 1e-3, len(valores))
        return lances[int(np.argmax(valores))]

    def aprender(self, exp):
        if len(exp) < MIN_TREINO:
            return None
        erros = []
        for _ in range(PASSOS_TREINO):
            X, _, y = exp.lote(LOTE)
            erros.append(self.rede.treinar(X, y))
        self.ultimo_erro = round(float(np.mean(erros[-10:])), 4)
        return self.ultimo_erro


class Boy(Agente):
    """Regressao sobre padroes. Explora com 'temperatura' (softmax) que diminui com o tempo."""

    def __init__(self, compartilhada):
        super().__init__("Boy", compartilhada)
        self.modelo = Regressao()
        self.temperatura = 0.08

    def escolher(self, board, antes, cor):
        lances, F, mate = self._avaliar_candidatos(board, antes, cor, feats_padroes)
        if mate:
            return mate
        valores = self.modelo.prever(F) + self.bonus_compartilhada(antes, lances)
        z = (valores - valores.max()) / max(self.temperatura, 1e-3)
        pesos = np.exp(z)
        return random.choices(lances, weights=pesos / pesos.sum(), k=1)[0]

    def aprender(self, exp):
        if len(exp) < MIN_TREINO:
            return None
        erros = []
        for _ in range(PASSOS_TREINO):
            _, F, y = exp.lote(LOTE)
            erros.append(self.modelo.treinar(F, y))
        self.ultimo_erro = round(float(np.mean(erros[-10:])), 4)
        return self.ultimo_erro


# ---------------------------------------------------------------------------
# Teclado (Windows e Linux/Mac)
# ---------------------------------------------------------------------------
class Teclado:
    def __init__(self):
        self.ativo = sys.stdin.isatty()
        self.windows = os.name == "nt"

    def __enter__(self):
        if self.ativo and not self.windows:
            import termios
            import tty
            self._termios = termios
            self.fd = sys.stdin.fileno()
            self.antigo = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)
        return self

    def __exit__(self, *args):
        if self.ativo and not self.windows:
            self._termios.tcsetattr(self.fd, self._termios.TCSADRAIN, self.antigo)

    def tecla(self):
        """Le uma tecla sem travar. Retorna None se nao houver nenhuma."""
        if not self.ativo:
            return None
        if self.windows:
            import msvcrt
            if msvcrt.kbhit():
                ch = msvcrt.getwch()
                if ch in ("\x00", "\xe0"):
                    msvcrt.getwch()
                    return None
                return ch.lower()
            return None
        import select
        if select.select([sys.stdin], [], [], 0)[0]:
            return os.read(self.fd, 1).decode(errors="ignore").lower()
        return None

    def esperar(self):
        """Espera uma tecla (sem terminal interativo, espera Enter)."""
        if not self.ativo:
            input()
            return " "
        while True:
            t = self.tecla()
            if t:
                return t
            time.sleep(0.03)


class Sair(Exception):
    pass


class Estado:
    def __init__(self, modo):
        self.modo = modo

    def tratar(self, t):
        if t in TECLAS_MODO:
            novo = TECLAS_MODO[t]
            if novo != self.modo:
                self.modo = novo
                print(f"\n>> Modo: {NOMES_MODO[novo]}")
        elif t == " " and self.modo in ("auto", "rapido"):
            self.modo = "lance"
            print(f"\n>> Pausado. Modo: {NOMES_MODO['lance']}")
        elif t == "q":
            raise Sair()


def esperar_espaco(tk, estado, modo_alvo, mensagem):
    print(mensagem)
    while estado.modo == modo_alvo:
        t = tk.esperar()
        if t in (" ", "\r", "\n"):
            return
        estado.tratar(t)


def passo(tk, estado):
    """Chamado depois de cada lance."""
    if estado.modo == "lance":
        esperar_espaco(tk, estado, "lance",
                       "   [ESPACO] proximo lance | A/R/L/P muda o modo | Q sai")
        return
    t = tk.tecla()
    while t:
        estado.tratar(t)
        t = tk.tecla()
    if estado.modo == "auto":
        time.sleep(PAUSA_AUTO)


# ---------------------------------------------------------------------------
# Excel
# ---------------------------------------------------------------------------
def _ler_aba(wb, nome):
    if nome not in wb.sheetnames:
        return []
    return [r for r in wb[nome].iter_rows(min_row=2, values_only=True)
            if r and r[0] is not None]


def carregar_tudo():
    comp = Compartilhada()
    hero, boy, exp = Hero(comp), Boy(comp), Experiencias()
    partidas = []
    if os.path.exists(ARQUIVO):
        try:
            wb = load_workbook(ARQUIVO, read_only=True, data_only=True)
            comp.carregar(_ler_aba(wb, "Compartilhada"))
            partidas = [tuple(r)[:11] for r in _ler_aba(wb, "Partidas")]

            w1 = [[float(v) for v in r[1:1 + OCULTA]] for r in _ler_aba(wb, "Hero_Rede")]
            resto = {r[0]: [float(v) for v in r[1:] if v is not None] for r in _ler_aba(wb, "Hero_Resto")}
            if (len(w1) == N_ENTRADA and all(len(l) == OCULTA for l in w1)
                    and len(resto.get("b1", [])) == OCULTA and len(resto.get("W2", [])) == OCULTA
                    and len(resto.get("b2", [])) == 1):
                hero.rede.W1[:] = np.array(w1)
                hero.rede.b1[:] = np.array(resto["b1"])
                hero.rede.W2[:, 0] = np.array(resto["W2"])
                hero.rede.b2[:] = np.array(resto["b2"])

            pesos = [float(r[1]) for r in _ler_aba(wb, "Boy_Modelo")]
            if len(pesos) == N_PADROES:
                boy.modelo.w[:] = np.array(pesos)

            exp.carregar(_ler_aba(wb, "Experiencias"))
            wb.close()
        except Exception as erro:  # arquivo corrompido, formato errado etc.
            backup = ARQUIVO + ".bak"
            try:
                os.replace(ARQUIVO, backup)
            except OSError:
                pass
            print(f"[aviso] Nao consegui ler o Excel ({erro}). Guardei o arquivo antigo em {backup} "
                  f"e vou comecar do zero.")
            comp = Compartilhada()
            hero, boy, exp = Hero(comp), Boy(comp), Experiencias()
            partidas = []
    return hero, boy, comp, exp, partidas


def _aba(wb, nome, cabecalho, linhas, larguras, primeira=False):
    ws = wb.active if primeira else wb.create_sheet()
    ws.title = nome
    ws.append(cabecalho)
    for celula in ws[1]:
        celula.font = Font(bold=True)
    for linha in linhas:
        ws.append(list(linha))
    ws.freeze_panes = "A2"
    for i, largura in enumerate(larguras, 1):
        ws.column_dimensions[get_column_letter(i)].width = largura
    return ws


def _nomes_entradas():
    nomes = []
    for grupo in range(12):
        dono = "minha" if grupo < 6 else "adversario"
        for casa in range(64):
            nomes.append(f"{dono} {NOMES_PECAS[grupo % 6]} em {chess.square_name(casa)}")
    nomes.append("saldo de material")
    return nomes


def salvar_excel(hero, boy, comp, exp, partidas):
    """Retorna True se salvou. Se o Excel estiver aberto, retorna False."""
    wb = Workbook()

    total = len(partidas)
    v_hero = sum(1 for p in partidas if p[5] == "Hero")
    v_boy = sum(1 for p in partidas if p[5] == "Boy")
    placar = [
        ("Total de partidas", total),
        ("Vitorias do Hero (rede neural)", v_hero),
        ("Vitorias do Boy (regressao)", v_boy),
        ("Empates", total - v_hero - v_boy),
        ("Vitorias do Hero jogando de brancas", sum(1 for p in partidas if p[2] == "Hero" and p[5] == "Hero")),
        ("Vitorias do Boy jogando de brancas", sum(1 for p in partidas if p[2] == "Boy" and p[5] == "Boy")),
        ("Experiencias no dataset de treino", len(exp)),
        ("Posicoes na tabela compartilhada", len(comp.dados)),
        ("Erro atual do Hero (menor = preve melhor)", hero.ultimo_erro),
        ("Erro atual do Boy (menor = preve melhor)", boy.ultimo_erro),
    ]
    ws_placar = _aba(wb, "Placar", ["Item", "Valor"], placar, [44, 12], primeira=True)

    ws_partidas = _aba(
        wb, "Partidas",
        ["N", "Data/Hora", "Brancas", "Pretas", "Resultado", "Vencedor", "Motivo",
         "Meio-lances", "Erro Hero", "Erro Boy", "Lances"],
        partidas, [6, 20, 10, 10, 11, 10, 38, 12, 10, 10, 100])

    if total >= 2:  # curva de aprendizado: o erro deve cair com o tempo
        grafico = LineChart()
        grafico.title = "Curva de aprendizado (erro de previsao)"
        grafico.y_axis.title = "erro"
        grafico.x_axis.title = "partida"
        grafico.add_data(Reference(ws_partidas, min_col=9, max_col=10, min_row=1, max_row=total + 1),
                         titles_from_data=True)
        grafico.set_categories(Reference(ws_partidas, min_col=1, min_row=2, max_row=total + 1))
        grafico.width, grafico.height = 20, 9
        ws_placar.add_chart(grafico, "D2")

    nomes = _nomes_entradas()
    w1 = ([nomes[i]] + [round(float(v), 6) for v in hero.rede.W1[i]] for i in range(N_ENTRADA))
    _aba(wb, "Hero_Rede", ["entrada"] + [f"neuronio {j + 1}" for j in range(OCULTA)],
         w1, [34] + [11] * OCULTA)
    resto = [["b1"] + [round(float(v), 6) for v in hero.rede.b1],
             ["W2"] + [round(float(v), 6) for v in hero.rede.W2[:, 0]],
             ["b2", round(float(hero.rede.b2[0]), 6)]]
    _aba(wb, "Hero_Resto", ["parametro"] + [f"neuronio {j + 1}" for j in range(OCULTA)],
         resto, [12] + [11] * OCULTA)

    _aba(wb, "Boy_Modelo", ["Padrao", "Peso aprendido"],
         [(n, round(float(w), 6)) for n, w in zip(NOMES_PADROES, boy.modelo.w)], [32, 16])
    _aba(wb, "Experiencias", ["Posicao apos o lance (EPD)", "Quem jogou (B/P)", "Alvo (-1 a +1)", "Partida"],
         exp.linhas(), [70, 16, 14, 10])
    _aba(wb, "Compartilhada", ["Posicao (EPD)", "Lance", "Vitorias", "Empates", "Derrotas"],
         comp.linhas(), [70, 10, 10, 10, 10])

    temporario = ARQUIVO.replace(".xlsx", "_tmp.xlsx")
    wb.save(temporario)
    try:
        os.replace(temporario, ARQUIVO)
    except PermissionError:
        print("\n[aviso] Nao consegui salvar: o Excel esta com memoria_xadrez_ml.xlsx aberto. "
              "Feche o arquivo.")
        return False
    return True


# ---------------------------------------------------------------------------
# Partida
# ---------------------------------------------------------------------------
def resultado_final(board):
    """Retorna (resultado do ponto de vista das brancas, motivo)."""
    out = board.outcome(claim_draw=True)
    if out:
        if out.winner is None:
            return 0, out.termination.name
        return (1 if out.winner else -1), out.termination.name
    dif = material(board, chess.WHITE)
    if dif >= 3:
        return 1, "LIMITE_MATERIAL"
    if dif <= -3:
        return -1, "LIMITE_MATERIAL"
    return 0, "LIMITE_EMPATE"


def jogar_partida(numero, brancas, pretas, tk, estado):
    board = chess.Board()
    brancas.novo_jogo()
    pretas.novo_jogo()

    if estado.modo in ("auto", "lance"):
        print(f"\n{'=' * 44}\n PARTIDA {numero}: {brancas.nome} (brancas) x {pretas.nome} (pretas)"
              f"\n{'=' * 44}")
        print(desenhar(board))
        if estado.modo == "lance":
            esperar_espaco(tk, estado, "lance", "   [ESPACO] comecar | A/R/L/P muda o modo | Q sai")

    while board.ply() < MAX_LANCES and not board.is_game_over(claim_draw=True):
        agente = brancas if board.turn == chess.WHITE else pretas
        lance = agente.jogar(board)
        san = board.san(lance)
        board.push(lance)
        if estado.modo in ("auto", "lance"):
            print(f"\n[{agente.nome}] {san}   (meio-lance {board.ply()})")
            print(desenhar(board))
        passo(tk, estado)

    res_brancas, motivo = resultado_final(board)
    lances_txt = chess.Board().variation_san(board.move_stack)
    return board, res_brancas, motivo, lances_txt


def main():
    print("=" * 52)
    print("  XADREZ: HERO (rede neural) x BOY (regressao)")
    print("=" * 52)
    hero, boy, comp, exp, partidas = carregar_tudo()
    print(f"Excel: {ARQUIVO}")
    print(f"Partidas ja jogadas: {len(partidas)} | experiencias no dataset: {len(exp)} | "
          f"tabela compartilhada: {len(comp.dados)} posicoes\n")

    print("Modo inicial:")
    print("  [A] automatico   [R] rapido   [L] ESPACO a cada lance   [P] ESPACO a cada partida")
    escolha = ""
    while escolha not in TECLAS_MODO:
        escolha = input("Escolha (A/R/L/P): ").strip().lower()[:1]
    entrada = input("Quantas partidas? (Enter = ate apertar Q): ").strip()
    total = int(entrada) if entrada.isdigit() and int(entrada) > 0 else None
    estado = Estado(TECLAS_MODO[escolha])

    feitas = 0
    print("\nDurante o jogo:  A/R/L/P = muda o modo | ESPACO = avanca/pausa | Q = sair\n")

    with Teclado() as tk:
        try:
            while total is None or feitas < total:
                # exploracao diminui conforme eles ganham experiencia
                fator = 0.99 ** len(partidas)
                hero.epsilon = max(0.03, 0.20 * fator)
                boy.temperatura = max(0.02, 0.08 * fator)

                # alterna quem joga de brancas (le a ultima partida do Excel)
                hero_brancas = partidas[-1][2] != "Hero" if partidas else True
                brancas, pretas = (hero, boy) if hero_brancas else (boy, hero)

                numero = len(partidas) + 1
                board, rb, motivo, lances_txt = jogar_partida(numero, brancas, pretas, tk, estado)

                # APRENDIZADO ---------------------------------------------
                for agente, res in ((brancas, rb), (pretas, -rb)):
                    n = len(agente.historico)
                    idx = {1: 0, 0: 1, -1: 2}[res]
                    for i, (antes, uci, depois, cor, x, f) in enumerate(agente.historico):
                        alvo = res * GAMMA ** (n - 1 - i)
                        exp.adicionar(depois, cor, alvo, x, f, numero)   # dataset compartilhado
                        comp.registrar(antes, uci, idx)                  # tabela compartilhada
                erro_hero = hero.aprender(exp)
                erro_boy = boy.aprender(exp)
                # ---------------------------------------------------------

                if rb == 1:
                    vencedor, texto = brancas.nome, "1-0"
                elif rb == -1:
                    vencedor, texto = pretas.nome, "0-1"
                else:
                    vencedor, texto = "Empate", "1/2-1/2"

                partidas.append((numero, datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                 brancas.nome, pretas.nome, texto, vencedor,
                                 MOTIVOS.get(motivo, motivo.lower()), board.ply(),
                                 erro_hero, erro_boy, lances_txt))
                feitas += 1

                resumo = "empate" if vencedor == "Empate" else f"venceu: {vencedor}"
                print(f"\n>>> Partida {numero}: {brancas.nome}(B) x {pretas.nome}(P)  {texto}  {resumo}  "
                      f"[{MOTIVOS.get(motivo, motivo)} | {board.ply()} meio-lances]")
                if erro_hero is not None:
                    print(f"    aprendizado -> erro Hero: {erro_hero} | erro Boy: {erro_boy} | "
                          f"exploracao Hero: {hero.epsilon:.0%} | dataset: {len(exp)}")

                if estado.modo != "rapido" or feitas % 10 == 0:
                    salvar_excel(hero, boy, comp, exp, partidas)

                if estado.modo == "partida":
                    print(desenhar(board))
                    esperar_espaco(tk, estado, "partida",
                                   "   [ESPACO] proxima partida | A/R/L/P muda o modo | Q sai")
                elif estado.modo == "auto":
                    time.sleep(1.0)
        except (Sair, KeyboardInterrupt):
            print("\nEncerrando...")

    while not salvar_excel(hero, boy, comp, exp, partidas):
        r = input("Feche o Excel e aperte Enter para tentar de novo (ou S para sair sem salvar): ")
        if r.strip().lower() == "s":
            break

    v_hero = sum(1 for p in partidas if p[5] == "Hero")
    v_boy = sum(1 for p in partidas if p[5] == "Boy")
    print("\n=== PLACAR GERAL ===")
    print(f"Hero: {v_hero} | Boy: {v_boy} | Empates: {len(partidas) - v_hero - v_boy} "
          f"| Total: {len(partidas)}")
    print("\nO que o Boy aprendeu (peso de cada padrao):")
    for nome, peso in zip(NOMES_PADROES, boy.modelo.w):
        print(f"  {nome:<28} {peso:+.3f}")
    print(f"\nSalvo em: {ARQUIVO}")


if __name__ == "__main__":
    main()
