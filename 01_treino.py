from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

PASTA_ALGORITMO = Path("/content/drive/MyDrive/Algoritmo")


def localizar_database() -> Path:
    if not PASTA_ALGORITMO.is_dir():
        raise FileNotFoundError(f"Pasta dos WAVs não encontrada: {PASTA_ALGORITMO}")
    return PASTA_ALGORITMO


def localizar_saida() -> Path:
    pasta = localizar_database() / "Algoritmo_saida"
    pasta.mkdir(parents=True, exist_ok=True)
    return pasta


def localizar_scripts() -> Path:
    return localizar_database()


import joblib
import numpy as np
import pandas as pd
import pywt
import sklearn
import soundfile as sf
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# C1 usa o sinal recortado; C2 inclui filtragem por wavelet antes da extração A3+B3.
PROCESSAMENTOS = ("C1", "C2")
USUARIOS = tuple("ABCDEFGHIJ")
N_TREINO_POR_USUARIO = 13
N_TREINO_TOTAL = len(USUARIOS) * N_TREINO_POR_USUARIO
FS = 44100
MIN_N = 64
WAVELET = "db4"
DWT_LEVEL = 1
DWT_MODE = "symmetric"
FATOR_DWT = 0.10
MAD_CONST = 0.6744897501960817
PERCENTUAIS = np.arange(5, 100, 5, dtype=float)
COLS_FEATURES = [f"A3_{int(q):02d}" for q in PERCENTUAIS] + [f"B3_{int(q):02d}" for q in PERCENTUAIS]
RANDOM_STATE = 42
INNER_SPLITS = 5
SVM_C = 20.0
RF_N_ESTIMATORS = 100
ALPHA_FUSAO = 0.75
FAR_ALVO_D = 0.01
METODOS = ("SVM_RBF", "RF", "FUSAO_SVM_RF")
N_METODOS = len(METODOS)
OUTER_FOLDS_POR_MODELO = N_TREINO_TOTAL
TOTAL_OUTER_FOLDS = len(PROCESSAMENTOS) * len(USUARIOS) * OUTER_FOLDS_POR_MODELO
TOTAL_LINHAS_NESTED = TOTAL_OUTER_FOLDS * N_METODOS
CSV_KW = dict(index=False, encoding="utf-8-sig", float_format="%.10g")



BASE_DIR = localizar_database()
AUDIO_ROOT = BASE_DIR
SAIDA_DIR = localizar_saida()
SCRIPTS_DIR = localizar_scripts()
MODELOS_DIR = SAIDA_DIR / "modelos"
RELATORIOS_DIR = SAIDA_DIR / "relatorios"
CACHE_DIR = SAIDA_DIR / "cache"
CHECKPOINT_DIR = SAIDA_DIR / "checkpoints"
CHECKPOINT_INCOMP_DIR = SAIDA_DIR / "checkpoints_incompativeis"
for d in (SAIDA_DIR, MODELOS_DIR, RELATORIOS_DIR, CACHE_DIR, CHECKPOINT_DIR, CHECKPOINT_INCOMP_DIR):
    d.mkdir(parents=True, exist_ok=True)


# A configuração das características é identificada por hash para validar cache e modelos.
def feature_config(proc: str) -> dict:
    if proc not in PROCESSAMENTOS:
        raise ValueError(proc)
    return {
        "schema": 1,
        "processamento": proc,
        "usa_dwt": proc == "C2",
        "fs_hz": FS,
        "min_n": MIN_N,
        "remocao_media_antes_recorte": True,
        "recorte_energia_inicio": 0.02,
        "recorte_energia_fim": 0.98,
        "wavelet": WAVELET if proc == "C2" else None,
        "dwt_level": DWT_LEVEL if proc == "C2" else None,
        "dwt_mode": DWT_MODE if proc == "C2" else None,
        "fator_dwt": FATOR_DWT if proc == "C2" else None,
        "mad_const": MAD_CONST if proc == "C2" else None,
        "threshold_mode": "soft" if proc == "C2" else None,
        "percentuais_a3_b3": [int(v) for v in PERCENTUAIS],
        "n_features": len(COLS_FEATURES),
        "politica_zeros_b3": "propagar_ultimo_sinal_nao_nulo; zeros_iniciais_recebem_primeiro_sinal_nao_nulo",
        "indice_b3": "idx+2_amostras",
    }


def hash_obj(obj) -> str:
    b = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(b).hexdigest()


FEATURE_CONFIGS = {p: feature_config(p) for p in PROCESSAMENTOS}
FEATURE_CONFIG_HASHES = {p: hash_obj(FEATURE_CONFIGS[p]) for p in PROCESSAMENTOS}
MODEL_CONFIG = {
    "svm": {"kernel": "rbf", "C": SVM_C, "gamma": "scale", "class_weight": "balanced", "standard_scaler": True},
    "rf": {"n_estimators": RF_N_ESTIMATORS, "class_weight": "balanced", "random_state": RANDOM_STATE, "n_jobs": -1},
    "fusao": {"alpha_svm": ALPHA_FUSAO, "peso_rf": 1.0 - ALPHA_FUSAO},
}
VALIDATION_CONFIG = {
    "outer": "LeaveOneOut",
    "inner": "StratifiedKFold",
    "inner_splits": INNER_SPLITS,
    "inner_shuffle": True,
    "inner_random_state": RANDOM_STATE,
    "politica_limiar": "D",
    "criterio_D": "maximizar_TAR_sujeito_a_FAR<=1%",
    "far_alvo_D": FAR_ALVO_D,
    "ordem_entrada": "numeracao_atual_de_arquivos_1_a_13",
}


def salvar_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, **CSV_KW)
    os.replace(tmp, path)


def salvar_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def salvar_joblib(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    joblib.dump(obj, tmp)
    os.replace(tmp, path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def validar_wav(path: Path):
    info = sf.info(str(path))
    erros = []
    if info.format != "WAV": erros.append(f"formato={info.format}")
    if info.subtype != "PCM_16": erros.append(f"subtype={info.subtype}")
    if info.channels != 1: erros.append(f"canais={info.channels}")
    if info.samplerate != FS: erros.append(f"fs={info.samplerate}")
    if info.frames < MIN_N: erros.append(f"frames={info.frames}")
    if erros: raise ValueError(f"{path.name}: " + ", ".join(erros))
    return info


def iter_wavs_treino():
    padrao = re.compile(r"^([A-J])_TREINO_(\d{2})\.wav$", re.I)
    for p in AUDIO_ROOT.iterdir():
        if not p.is_file() or p.suffix.lower() != ".wav":
            continue
        m = padrao.fullmatch(p.name)
        if m and 1 <= int(m.group(2)) <= N_TREINO_POR_USUARIO:
            yield p


# Ordena os WAVs por proprietário e numeração atual, validando formato e conteúdo único.
def construir_inventario_treino() -> pd.DataFrame:
    encontrados = {}
    for p in iter_wavs_treino():
        key = p.name.upper()
        if key in encontrados:
            raise RuntimeError(f"Nome duplicado: {p} / {encontrados[key]}")
        encontrados[key] = p

    rows = []
    hashes = {}
    for usuario in USUARIOS:
        for numero in range(1, N_TREINO_POR_USUARIO + 1):
            nome = f"{usuario}_TREINO_{numero:02d}.wav"
            p = encontrados.get(nome.upper())
            if p is None:
                raise FileNotFoundError(f"Arquivo de treinamento ausente: {nome}")
            info = validar_wav(p)
            sha = sha256(p)
            if sha in hashes:
                raise RuntimeError(f"Áudio duplicado: {p} / {hashes[sha]}")
            hashes[sha] = p
            rows.append({
                "Posicao_Global": len(rows) + 1,
                "Usuario": usuario,
                "Numero": numero,
                "Arquivo_Logico": nome,
                "Arquivo": p.name,
                "Caminho": str(p.resolve()),
                "SHA256": sha,
                "Frames": int(info.frames),
                "Duracao_s": float(info.duration),
                "Fs_Hz": int(info.samplerate),
            })
    df = pd.DataFrame(rows)
    salvar_csv(df, SAIDA_DIR / "inventario_treino.csv")
    return df


def fingerprint_ordem(df: pd.DataFrame) -> str:
    txt = "\n".join(f"{u}|{s}" for u, s in zip(df.Usuario.astype(str), df.SHA256.astype(str)))
    return hashlib.sha256(txt.encode("utf-8")).hexdigest()


def fingerprint_conteudo(df: pd.DataFrame) -> str:
    txt = "\n".join(sorted(f"{u}|{s}" for u, s in zip(df.Usuario.astype(str), df.SHA256.astype(str))))
    return hashlib.sha256(txt.encode("utf-8")).hexdigest()


# Lê áudio WAV PCM mono e remove a componente contínua antes do recorte energético.
def carregar_sinal(path: Path) -> np.ndarray:
    validar_wav(path)
    x, fs = sf.read(str(path), dtype="float64", always_2d=False)
    x = np.asarray(x, float)
    if fs != FS or x.ndim != 1 or x.size < MIN_N or not np.all(np.isfinite(x)):
        raise ValueError(path.name)
    x -= np.mean(x)
    if np.sum(x * x) <= np.finfo(float).eps: raise ValueError(f"{path.name}: sem energia")
    return x


# Delimita o trecho entre 2% e 98% da energia acumulada do sinal centralizado.
def recortar_energia(x: np.ndarray) -> np.ndarray:
    e = np.cumsum(x * x)
    total = float(e[-1])
    i = int(np.searchsorted(e, 0.02 * total, "left"))
    j = int(np.searchsorted(e, 0.98 * total, "left")) + 1
    y = x[i:min(j, len(x))]
    if y.size < MIN_N: raise ValueError("Recorte curto")
    return y


def sigma_mad(d: np.ndarray) -> float:
    s = np.median(np.abs(d - np.median(d))) / MAD_CONST
    if not np.isfinite(s) or s <= np.finfo(float).eps: s = np.std(d, ddof=1)
    if not np.isfinite(s) or s <= np.finfo(float).eps: raise ValueError("MAD inválido")
    return float(s)


# Em C2, atenua detalhes wavelet por limiarização suave e reconstrói o sinal.
def aplicar_dwt(x: np.ndarray) -> np.ndarray:
    a, d = pywt.wavedec(x, WAVELET, level=DWT_LEVEL, mode=DWT_MODE)
    lim = FATOR_DWT * sigma_mad(d) * np.sqrt(2.0 * np.log(max(len(d), 2)))
    y = pywt.waverec([a, pywt.threshold(d, lim, mode="soft")], WAVELET, mode=DWT_MODE)[:len(x)]
    y = np.asarray(y, float)
    y -= np.mean(y)
    return y


# A3: posições relativas dos percentuais acumulados de energia (5% a 95%).
def a3(x: np.ndarray) -> np.ndarray:
    e = np.cumsum(x * x); total = float(e[-1])
    idx = np.searchsorted(e, (PERCENTUAIS / 100.0) * total, "left")
    return np.minimum(idx + 1, len(x)) / len(x)


# B3: posições relativas dos percentuais acumulados de cruzamentos por zero.
def b3(x: np.ndarray) -> np.ndarray:
    s = np.sign(x - np.mean(x)).astype(np.int8)
    # Zeros interiores propagam o último sinal não nulo; zeros iniciais usam o primeiro sinal válido.
    nz = np.flatnonzero(s)
    if nz.size == 0: raise ValueError("B3 indefinida")
    s[:nz[0]] = s[nz[0]]
    for k in range(nz[0] + 1, len(s)):
        if s[k] == 0: s[k] = s[k - 1]
    z = (s[:-1] * s[1:] < 0).astype(int)
    total = int(z.sum())
    if total == 0: raise ValueError("B3 sem cruzamentos")
    c = np.cumsum(z)
    idx = np.searchsorted(c, (PERCENTUAIS / 100.0) * total, "left")
    return np.minimum(idx + 2, len(x)) / len(x)


def extrair_features(path: Path, proc: str) -> np.ndarray:
    x = recortar_energia(carregar_sinal(path))
    if proc == "C2": x = aplicar_dwt(x)
    elif proc != "C1": raise ValueError(proc)
    v = np.r_[a3(x), b3(x)]
    if v.shape != (38,) or not np.all(np.isfinite(v)): raise ValueError(path.name)
    return v


# Reaproveita as 38 características somente se os hashes dos WAVs e da configuração coincidirem.
def obter_features(inventario: pd.DataFrame, proc: str) -> np.ndarray:
    npz = CACHE_DIR / f"features_treino_{proc}.npz"
    csv = CACHE_DIR / f"features_treino_{proc}.csv"
    shas = inventario.SHA256.to_numpy(str)
    cfg_hash = FEATURE_CONFIG_HASHES[proc]
    if npz.exists():
        try:
            d = np.load(npz, allow_pickle=False)
            if d["X"].shape == (N_TREINO_TOTAL, 38) and np.array_equal(d["sha"], shas) and str(d["feature_config_hash"].item()) == cfg_hash:
                return d["X"]
        except Exception:
            pass
    X = np.stack([extrair_features(Path(p), proc) for p in inventario.Caminho])
    np.savez_compressed(npz, X=X, sha=shas, feature_config_hash=np.asarray(cfg_hash))
    df = pd.DataFrame(X, columns=COLS_FEATURES)
    df.insert(0, "Numero", inventario.Numero.to_numpy(int)); df.insert(0, "Arquivo", inventario.Arquivo.to_numpy(str)); df.insert(0, "Usuario", inventario.Usuario.to_numpy(str))
    salvar_csv(df, csv)
    return X


# Padronização e SVM de núcleo RBF são ajustadas exclusivamente nos dados de cada treino.
def criar_svm() -> Pipeline:
    return Pipeline([("escala", StandardScaler()), ("modelo", SVC(kernel="rbf", C=SVM_C, gamma="scale", class_weight="balanced"))])


# Random Forest usa pesos de classes balanceados e semente fixa para reprodutibilidade.
def criar_rf() -> Pipeline:
    return Pipeline([("modelo", RandomForestClassifier(n_estimators=RF_N_ESTIMATORS, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1))])


def score_svm(modelo, X): return np.asarray(modelo.decision_function(X), float).reshape(-1)

def score_rf(modelo, X):
    probs = modelo.predict_proba(X); classes = list(modelo.named_steps["modelo"].classes_)
    return np.asarray(probs[:, classes.index(1)], float)


# Produz escores fora da partição de treino, evitando calibrar no próprio ajuste do modelo.
def oof_duplo_5fold(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cv = StratifiedKFold(n_splits=INNER_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    s1 = np.full(len(y), np.nan); s2 = np.full(len(y), np.nan)
    for tr, va in cv.split(X, y):
        svm, rf = criar_svm(), criar_rf(); svm.fit(X[tr], y[tr]); rf.fit(X[tr], y[tr])
        s1[va] = score_svm(svm, X[va]); s2[va] = score_rf(rf, X[va])
    return s1, s2


def escala_robusta(x: np.ndarray) -> float:
    q25, q75 = np.quantile(x, [0.25, 0.75]); s = (q75 - q25) / 1.3489795003921634
    if not np.isfinite(s) or s <= np.finfo(float).eps: s = np.median(np.abs(x - np.median(x))) / MAD_CONST
    if not np.isfinite(s) or s <= np.finfo(float).eps: s = np.std(x, ddof=1)
    return float(s if np.isfinite(s) and s > np.finfo(float).eps else 1e-6)


# A normalização usa mediana e escala robusta dos escores de impostores.
def normalizar_scores(scores: np.ndarray, y: np.ndarray):
    imp = scores[y == 0]; med = float(np.median(imp)); esc = escala_robusta(imp)
    return (scores - med) / esc, med, esc


def estatisticas_limiar(scores: np.ndarray, y: np.ndarray, lim: float):
    ac = scores >= lim; g = y == 1; i = y == 0
    ga = int(np.sum(ac & g)); ia = int(np.sum(ac & i)); tar = ga / int(np.sum(g)); far = ia / int(np.sum(i))
    return ga, ia, tar, far


# Política D: maximiza TAR entre os limiares com FAR de calibração até 1%.
def melhor_limiar_d(scores: np.ndarray, y: np.ndarray) -> dict:
    vals = np.unique(np.asarray(scores, float))
    cand = np.unique(np.r_[np.nextafter(vals.min(), -np.inf), vals, np.nextafter(vals, np.inf)])
    validos = []
    for lim in cand:
        ga, ia, tar, far = estatisticas_limiar(scores, y, float(lim))
        if far <= FAR_ALVO_D + 1e-15:
            validos.append({"Limiar": float(lim), "GA": ga, "IA": ia, "TAR": tar, "FAR": far})
    if not validos: raise RuntimeError("Nenhum limiar D válido")
    validos.sort(key=lambda r: (r["TAR"], -r["IA"], r["Limiar"]), reverse=True)
    return validos[0]


def resultado_biometrico(classe: int, aceita: bool) -> str:
    if classe == 1: return "GA" if aceita else "GR"
    return "IA" if aceita else "IR"


def checkpoint_path(proc: str, usuario: str) -> Path:
    return CHECKPOINT_DIR / f"{proc}_{usuario}_nested.csv"


# Retoma somente dobras concluídas e compatíveis com o identificador experimental.
def carregar_checkpoint(proc: str, usuario: str, experimento_id: str):
    p = checkpoint_path(proc, usuario)
    if not p.exists(): return pd.DataFrame(), set()
    try: df = pd.read_csv(p)
    except Exception: return pd.DataFrame(), set()
    req = {"Experimento_ID", "Processamento", "Dobra_Externa", "Metodo"}
    if not req.issubset(df.columns) or set(df.Experimento_ID.astype(str)) != {experimento_id} or set(df.Processamento.astype(str)) != {proc}:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S"); shutil.move(str(p), str(CHECKPOINT_INCOMP_DIR / f"{p.stem}_{ts}.csv")); return pd.DataFrame(), set()
    cont = df.groupby("Dobra_Externa").Metodo.nunique(); completos = {int(k) for k, n in cont.items() if int(n) == N_METODOS}
    df = df[df.Dobra_Externa.astype(int).isin(completos)].copy()
    return df, completos


# LOOCV externo por proprietário; cada dobra recalibra escores e limiares no subconjunto de treino.
def nested_usuario(proc: str, usuario: str, X: np.ndarray, inventario: pd.DataFrame, experimento_id: str, parcial: pd.DataFrame, completos: set[int]) -> pd.DataFrame:
    y = (inventario.Usuario.to_numpy(str) == usuario).astype(int)
    outer = LeaveOneOut(); p = checkpoint_path(proc, usuario)
    for dobra, (tr, te) in enumerate(outer.split(X, y), start=1):
        if dobra in completos: continue
        ytr = y[tr]; k = int(te[0])
        ss, rr = oof_duplo_5fold(X[tr], ytr)
        zs, ms, es = normalizar_scores(ss, ytr); zr, mr, er = normalizar_scores(rr, ytr)
        zf = ALPHA_FUSAO * zs + (1 - ALPHA_FUSAO) * zr
        ls, lr, lf = melhor_limiar_d(zs, ytr), melhor_limiar_d(zr, ytr), melhor_limiar_d(zf, ytr)
        svm, rf = criar_svm(), criar_rf(); svm.fit(X[tr], ytr); rf.fit(X[tr], ytr)
        raw_s = float(score_svm(svm, X[te])[0]); raw_r = float(score_rf(rf, X[te])[0])
        z_s = (raw_s - ms) / es; z_r = (raw_r - mr) / er; z_f = ALPHA_FUSAO * z_s + (1 - ALPHA_FUSAO) * z_r
        classe = int(y[k]); r = inventario.iloc[k]
        novas = []
        for metodo, sc, li in (("SVM_RBF", z_s, ls["Limiar"]), ("RF", z_r, lr["Limiar"]), ("FUSAO_SVM_RF", z_f, lf["Limiar"])):
            ac = bool(sc >= li)
            novas.append({"Experimento_ID": experimento_id, "Processamento": proc, "Politica": "D", "Metodo": metodo, "Modelo_Dono": usuario, "Usuario_Real": str(r.Usuario), "Arquivo": str(r.Arquivo), "Numero": int(r.Numero), "Dobra_Externa": dobra, "Classe_Real": classe, "Score_Z": float(sc), "Limiar_Z": float(li), "Decisao": "ACEITA" if ac else "REJEITA", "Resultado": resultado_biometrico(classe, ac)})
        parcial = pd.concat([parcial, pd.DataFrame(novas)], ignore_index=True)
        salvar_csv(parcial.sort_values(["Dobra_Externa", "Metodo"]), p); completos.add(dobra)
        print(f"\r[{proc}/{usuario}] outer {dobra:3d}/130", end="", flush=True)
    print()
    return parcial.reset_index(drop=True)


# Recalibra por proprietário em dados de desenvolvimento e salva os modelos definitivos.
def calibrar_final(proc: str, usuario: str, X: np.ndarray, inventario: pd.DataFrame, experimento_id: str, dataset_fp: str) -> dict:
    y = (inventario.Usuario.to_numpy(str) == usuario).astype(int)
    ss, rr = oof_duplo_5fold(X, y)
    zs, ms, es = normalizar_scores(ss, y); zr, mr, er = normalizar_scores(rr, y); zf = ALPHA_FUSAO * zs + (1 - ALPHA_FUSAO) * zr
    ls, lr, lf = melhor_limiar_d(zs, y), melhor_limiar_d(zr, y), melhor_limiar_d(zf, y)
    svm, rf = criar_svm(), criar_rf(); svm.fit(X, y); rf.fit(X, y)
    pasta = MODELOS_DIR / proc / usuario; pasta.mkdir(parents=True, exist_ok=True)
    salvar_joblib(svm, pasta / "svm_rbf.joblib"); salvar_joblib(rf, pasta / "random_forest.joblib")
    cfg = {
        "Experimento_ID": experimento_id, "Dataset_Fingerprint": dataset_fp, "Processamento": proc, "Politica": "D",
        "Feature_Config_Hash": FEATURE_CONFIG_HASHES[proc],
        "SVM": {"mediana_impostor_oof": ms, "escala_robusta_impostor_oof": es, "limiar_z": float(ls["Limiar"])},
        "RF": {"mediana_impostor_oof": mr, "escala_robusta_impostor_oof": er, "limiar_z": float(lr["Limiar"])},
        "FUSAO": {"alpha": ALPHA_FUSAO, "peso_svm": ALPHA_FUSAO, "peso_rf": 1 - ALPHA_FUSAO, "limiar_z": float(lf["Limiar"])},
    }
    salvar_json(cfg, pasta / "configuracao.json")
    return {"Processamento": proc, "Usuario": usuario, "Limiar_SVM_Z": ls["Limiar"], "Limiar_RF_Z": lr["Limiar"], "Limiar_Fusao_Z": lf["Limiar"], "GA_SVM_Calibracao": ls["GA"], "IA_SVM_Calibracao": ls["IA"], "GA_RF_Calibracao": lr["GA"], "IA_RF_Calibracao": lr["IA"], "GA_Fusao_Calibracao": lf["GA"], "IA_Fusao_Calibracao": lf["IA"]}


def versoes_ambiente() -> dict:
    return {"python": sys.version.split()[0], "platform": platform.platform(), "numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__, "pywavelets": pywt.__version__, "soundfile": sf.__version__, "joblib": joblib.__version__}


REGEX = re.compile(r"^([A-J])_(TREINO|EXT)_(\d{2})\.wav$", re.IGNORECASE)


# Verifica a base completa antes de iniciar a validação aninhada, incluindo ausência de duplicatas.
def validar_amostras() -> pd.DataFrame:
    db = localizar_database()
    if not db.is_dir():
        raise FileNotFoundError(f"Pasta de amostras não encontrada: {db}")
    resultados = {}
    ambiguidades = []
    for p in db.iterdir():
        if not p.is_file() or p.suffix.lower() != ".wav":
            continue
        m = REGEX.fullmatch(p.name)
        if m is None:
            ambiguidades.append(str(p.relative_to(db)))
            continue
        usuario, grupo, numero = m.group(1).upper(), m.group(2).upper(), int(m.group(3))
        esperado = "TREINO" if 1 <= numero <= 13 else "EXT" if 14 <= numero <= 26 else None
        if esperado != grupo:
            ambiguidades.append(str(p.relative_to(db)))
            continue
        key = (usuario, grupo, numero)
        if key in resultados:
            raise RuntimeError(f"Nome lógico repetido: {key}: {p} / {resultados[key]}")
        resultados[key] = p

    if ambiguidades:
        raise ValueError(f"Arquivos WAV fora da nomenclatura acordada: {ambiguidades[:20]}")
    faltantes = [
        (usuario, "TREINO" if numero <= 13 else "EXT", numero)
        for usuario in USUARIOS for numero in range(1, 27)
        if (usuario, "TREINO" if numero <= 13 else "EXT", numero) not in resultados
    ]
    if faltantes:
        raise FileNotFoundError(f"Faltam {len(faltantes)} gravações. Exemplos: {faltantes[:15]}")

    rows = []
    hashes = {}
    for usuario in USUARIOS:
        for numero in range(1, 27):
            grupo = "TREINO" if numero <= 13 else "EXT"
            path = resultados[(usuario, grupo, numero)]
            info = sf.info(path)
            if (info.format, info.subtype, info.channels, info.samplerate) != ("WAV", "PCM_16", 1, 44100) or info.frames < 64:
                raise ValueError(f"Formato não compatível: {path.name}: {info}")
            digest = sha256(path)
            if digest in hashes:
                raise RuntimeError(f"Gravação duplicada por SHA256: {path} / {hashes[digest]}")
            hashes[digest] = path
            rows.append({
                "Usuario": usuario,
                "Numero_Atual": numero,
                "Grupo_Atual": grupo,
                "Arquivo_Atual": path.name,
                "SHA256": digest,
                "Duracao_s": float(info.duration),
                "Frames": int(info.frames),
            })

    df = pd.DataFrame(rows)
    destino = localizar_saida() / "auditoria"
    destino.mkdir(parents=True, exist_ok=True)
    df.to_csv(destino / "inventario_por_nome_atual.csv", index=False, encoding="utf-8-sig")
    relatorio = {
        "criterio_efetivo": "arquivos atuais numerados 01-13 e 14-26, respectivamente",
        "ordem_cronologica_de_coleta_confirmada": False,
        "quantidade_wav": int(len(df)),
        "treinamento": int((df.Grupo_Atual == "TREINO").sum()),
        "externo": int((df.Grupo_Atual == "EXT").sum()),
        "hashes_distintos": int(df.SHA256.nunique()),
    }
    (destino / "validacao_base.json").write_text(json.dumps(relatorio, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Base validada: {len(df)} gravações; 130 de treinamento, 130 externas; hashes distintos.")
    print(f"Inventário: {destino / 'inventario_por_nome_atual.csv'}")
    return df



# Executa C1 e C2, registra predições fora da amostra, calibra e persiste os modelos.
def main() -> None:
    ini = time.perf_counter()
    validar_amostras()
    inventario = construir_inventario_treino(); dataset_fp = fingerprint_ordem(inventario); content_fp = fingerprint_conteudo(inventario)
    experimento_id = hash_obj({"dataset": dataset_fp, "features": FEATURE_CONFIG_HASHES, "modelos": MODEL_CONFIG, "validacao": VALIDATION_CONFIG})
    salvar_json({"Estado": "EM_EXECUCAO", "Processamentos": list(PROCESSAMENTOS), "Politica": "D", "Amostras_Treino": N_TREINO_TOTAL, "Criterio_Selecao": "numeracao_atual_dos_arquivos; nao_atesta_ordem_cronologica_de_coleta", "Feature_Config": FEATURE_CONFIGS, "Feature_Config_Hashes": FEATURE_CONFIG_HASHES, "Model_Config": MODEL_CONFIG, "Validation_Config": VALIDATION_CONFIG, "Dataset_Fingerprint": dataset_fp, "Content_Fingerprint": content_fp, "Experimento_ID": experimento_id, "Ambiente": versoes_ambiente()}, SAIDA_DIR / "configuracao_global.json")
    Xs = {p: obter_features(inventario, p) for p in PROCESSAMENTOS}
    blocos = []
    for proc in PROCESSAMENTOS:
        for usuario in USUARIOS:
            parcial, completos = carregar_checkpoint(proc, usuario, experimento_id)
            blocos.append(nested_usuario(proc, usuario, Xs[proc], inventario, experimento_id, parcial, completos))
    nested = pd.concat(blocos, ignore_index=True)
    if len(nested) != TOTAL_LINHAS_NESTED: raise AssertionError(f"Nested: {len(nested)} != {TOTAL_LINHAS_NESTED}")
    salvar_csv(nested, RELATORIOS_DIR / "01_resultados_CV_NESTED_detalhado.csv")
    resumo = []
    for proc in PROCESSAMENTOS:
        for usuario in USUARIOS:
            resumo.append(calibrar_final(proc, usuario, Xs[proc], inventario, experimento_id, dataset_fp))
    salvar_csv(pd.DataFrame(resumo), RELATORIOS_DIR / "02_calibracao_final_por_usuario.csv")
    tempo = time.perf_counter() - ini
    salvar_json({"Estado": "TREINAMENTO_CONCLUIDO", "Processamentos": list(PROCESSAMENTOS), "Politica": "D", "Amostras_Treino": N_TREINO_TOTAL, "Criterio_Selecao": "numeracao_atual_dos_arquivos; nao_atesta_ordem_cronologica_de_coleta", "Feature_Config": FEATURE_CONFIGS, "Feature_Config_Hashes": FEATURE_CONFIG_HASHES, "Model_Config": MODEL_CONFIG, "Validation_Config": VALIDATION_CONFIG, "Dataset_Fingerprint": dataset_fp, "Content_Fingerprint": content_fp, "Experimento_ID": experimento_id, "Ambiente": versoes_ambiente(), "Tempo_Total_s": tempo}, SAIDA_DIR / "configuracao_global.json")
    print(f"Concluído. Saída: {SAIDA_DIR}")


if __name__ == "__main__":
    main()