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

VERSAO = "5.7"
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


def _ancestral_db(path: Path) -> Path | None:
    path = path.resolve()
    for p in [path] + list(path.parents):
        if p.name.lower() == "db" and p.is_dir():
            return p
    return None


def resolver_base_dir() -> Path:
    env = os.environ.get("BIOMETRIA_DB")
    if env:
        p = Path(env).expanduser().resolve()
        if p.is_dir():
            return p
        raise FileNotFoundError(p)
    if "__file__" in globals():
        p = _ancestral_db(Path(__file__).resolve().parent)
        if p is not None:
            return p
    p = _ancestral_db(Path.cwd())
    if p is not None:
        return p
    colab = Path("/content/drive/MyDrive/DB")
    if colab.is_dir():
        return colab.resolve()
    raise FileNotFoundError("Pasta DB não localizada.")


BASE_DIR = resolver_base_dir()
AUDIO_ROOT = BASE_DIR
SAIDA_DIR = BASE_DIR / "Algoritmo_v5.7"
MODELOS_DIR = SAIDA_DIR / "modelos"
RELATORIOS_DIR = SAIDA_DIR / "relatorios"
CACHE_DIR = SAIDA_DIR / "cache"
CHECKPOINT_DIR = SAIDA_DIR / "checkpoints"
CHECKPOINT_INCOMP_DIR = SAIDA_DIR / "checkpoints_incompativeis"
for d in (SAIDA_DIR, MODELOS_DIR, RELATORIOS_DIR, CACHE_DIR, CHECKPOINT_DIR, CHECKPOINT_INCOMP_DIR):
    d.mkdir(parents=True, exist_ok=True)

MANIFESTO_TREINO = BASE_DIR / "manifesto_treino_v5_7.csv"


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
    "ordem_entrada": "manifesto_fixo_por_SHA256",
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


def ler_manifesto_fixo() -> pd.DataFrame:
    if not MANIFESTO_TREINO.exists():
        raise FileNotFoundError(f"Copie {MANIFESTO_TREINO.name} para a raiz de DB.")
    df = pd.read_csv(MANIFESTO_TREINO)
    req = {"Posicao_Global", "Usuario", "Posicao_Usuario", "Arquivo_Logico", "SHA256"}
    if not req.issubset(df.columns):
        raise AssertionError(f"Manifesto sem colunas: {sorted(req - set(df.columns))}")
    if len(df) != N_TREINO_TOTAL or df.SHA256.astype(str).duplicated().any():
        raise AssertionError("Manifesto de treino incompatível.")
    if not np.array_equal(df.Posicao_Global.astype(int).to_numpy(), np.arange(1, N_TREINO_TOTAL + 1)):
        raise AssertionError("Posicao_Global inválida.")
    esperado = np.repeat(np.asarray(USUARIOS, dtype=str), N_TREINO_POR_USUARIO)
    if not np.array_equal(df.Usuario.astype(str).to_numpy(), esperado):
        raise AssertionError("Ordem de usuários inválida no manifesto.")
    return df.reset_index(drop=True)


def iter_wavs_treino():
    padrao = re.compile(r"^[A-J]_TREINO_\d{2}\.wav$", re.I)
    for p in AUDIO_ROOT.rglob("*"):
        if not p.is_file() or not padrao.match(p.name):
            continue
        try:
            p.resolve().relative_to(SAIDA_DIR.resolve())
            continue
        except ValueError:
            pass
        yield p


def construir_manifesto_treino() -> pd.DataFrame:
    fixo = ler_manifesto_fixo()
    esperados = set(fixo.SHA256.astype(str))
    encontrados: dict[str, Path] = {}
    for p in iter_wavs_treino():
        s = sha256(p)
        if s in esperados:
            if s in encontrados:
                raise RuntimeError(f"Conteúdo duplicado: {p} e {encontrados[s]}")
            encontrados[s] = p
    faltam = esperados - set(encontrados)
    if faltam:
        raise FileNotFoundError(f"Faltam {len(faltam)} conteúdos do treino.")
    rows = []
    for r in fixo.itertuples(index=False):
        p = encontrados[str(r.SHA256)]
        info = validar_wav(p)
        rows.append({
            "Posicao_Global": int(r.Posicao_Global), "Usuario": str(r.Usuario), "Numero": int(r.Posicao_Usuario),
            "Arquivo_Logico": str(r.Arquivo_Logico), "Arquivo": p.name, "Caminho": str(p.resolve()), "SHA256": str(r.SHA256),
            "Frames": int(info.frames), "Duracao_s": float(info.duration), "Fs_Hz": int(info.samplerate),
        })
    df = pd.DataFrame(rows)
    salvar_csv(df, SAIDA_DIR / "manifesto_treino.csv")
    return df


def fingerprint_ordem(df: pd.DataFrame) -> str:
    txt = "\n".join(f"{u}|{s}" for u, s in zip(df.Usuario.astype(str), df.SHA256.astype(str)))
    return hashlib.sha256(txt.encode("utf-8")).hexdigest()


def fingerprint_conteudo(df: pd.DataFrame) -> str:
    txt = "\n".join(sorted(f"{u}|{s}" for u, s in zip(df.Usuario.astype(str), df.SHA256.astype(str))))
    return hashlib.sha256(txt.encode("utf-8")).hexdigest()


def carregar_sinal(path: Path) -> np.ndarray:
    validar_wav(path)
    x, fs = sf.read(str(path), dtype="float64", always_2d=False)
    x = np.asarray(x, float)
    if fs != FS or x.ndim != 1 or x.size < MIN_N or not np.all(np.isfinite(x)):
        raise ValueError(path.name)
    x -= np.mean(x)
    if np.sum(x * x) <= np.finfo(float).eps: raise ValueError(f"{path.name}: sem energia")
    return x


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


def aplicar_dwt(x: np.ndarray) -> np.ndarray:
    a, d = pywt.wavedec(x, WAVELET, level=DWT_LEVEL, mode=DWT_MODE)
    lim = FATOR_DWT * sigma_mad(d) * np.sqrt(2.0 * np.log(max(len(d), 2)))
    y = pywt.waverec([a, pywt.threshold(d, lim, mode="soft")], WAVELET, mode=DWT_MODE)[:len(x)]
    y = np.asarray(y, float)
    y -= np.mean(y)
    return y


def a3(x: np.ndarray) -> np.ndarray:
    e = np.cumsum(x * x); total = float(e[-1])
    idx = np.searchsorted(e, (PERCENTUAIS / 100.0) * total, "left")
    return np.minimum(idx + 1, len(x)) / len(x)


def b3(x: np.ndarray) -> np.ndarray:
    s = np.sign(x - np.mean(x)).astype(np.int8)
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


def obter_features(manifesto: pd.DataFrame, proc: str) -> np.ndarray:
    npz = CACHE_DIR / f"features_treino_{proc}.npz"
    csv = CACHE_DIR / f"features_treino_{proc}.csv"
    shas = manifesto.SHA256.to_numpy(str)
    cfg_hash = FEATURE_CONFIG_HASHES[proc]
    if npz.exists():
        try:
            d = np.load(npz, allow_pickle=False)
            if d["X"].shape == (N_TREINO_TOTAL, 38) and np.array_equal(d["sha"], shas) and str(d["feature_config_hash"].item()) == cfg_hash:
                return d["X"]
        except Exception:
            pass
    X = np.stack([extrair_features(Path(p), proc) for p in manifesto.Caminho])
    np.savez_compressed(npz, X=X, sha=shas, feature_config_hash=np.asarray(cfg_hash))
    df = pd.DataFrame(X, columns=COLS_FEATURES)
    df.insert(0, "Numero", manifesto.Numero.to_numpy(int)); df.insert(0, "Arquivo", manifesto.Arquivo.to_numpy(str)); df.insert(0, "Usuario", manifesto.Usuario.to_numpy(str))
    salvar_csv(df, csv)
    return X


def criar_svm() -> Pipeline:
    return Pipeline([("escala", StandardScaler()), ("modelo", SVC(kernel="rbf", C=SVM_C, gamma="scale", class_weight="balanced"))])


def criar_rf() -> Pipeline:
    return Pipeline([("modelo", RandomForestClassifier(n_estimators=RF_N_ESTIMATORS, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1))])


def score_svm(modelo, X): return np.asarray(modelo.decision_function(X), float).reshape(-1)

def score_rf(modelo, X):
    probs = modelo.predict_proba(X); classes = list(modelo.named_steps["modelo"].classes_)
    return np.asarray(probs[:, classes.index(1)], float)


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


def normalizar_scores(scores: np.ndarray, y: np.ndarray):
    imp = scores[y == 0]; med = float(np.median(imp)); esc = escala_robusta(imp)
    return (scores - med) / esc, med, esc


def estatisticas_limiar(scores: np.ndarray, y: np.ndarray, lim: float):
    ac = scores >= lim; g = y == 1; i = y == 0
    ga = int(np.sum(ac & g)); ia = int(np.sum(ac & i)); tar = ga / int(np.sum(g)); far = ia / int(np.sum(i))
    return ga, ia, tar, far


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


def nested_usuario(proc: str, usuario: str, X: np.ndarray, manifesto: pd.DataFrame, experimento_id: str, parcial: pd.DataFrame, completos: set[int]) -> pd.DataFrame:
    y = (manifesto.Usuario.to_numpy(str) == usuario).astype(int)
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
        classe = int(y[k]); r = manifesto.iloc[k]
        novas = []
        for metodo, sc, li in (("SVM_RBF", z_s, ls["Limiar"]), ("RF", z_r, lr["Limiar"]), ("FUSAO_SVM_RF", z_f, lf["Limiar"])):
            ac = bool(sc >= li)
            novas.append({"Experimento_ID": experimento_id, "Processamento": proc, "Politica": "D", "Metodo": metodo, "Modelo_Dono": usuario, "Usuario_Real": str(r.Usuario), "Arquivo": str(r.Arquivo), "Numero": int(r.Numero), "Dobra_Externa": dobra, "Classe_Real": classe, "Score_Z": float(sc), "Limiar_Z": float(li), "Decisao": "ACEITA" if ac else "REJEITA", "Resultado": resultado_biometrico(classe, ac)})
        parcial = pd.concat([parcial, pd.DataFrame(novas)], ignore_index=True)
        salvar_csv(parcial.sort_values(["Dobra_Externa", "Metodo"]), p); completos.add(dobra)
        print(f"\r[{proc}/{usuario}] outer {dobra:3d}/130", end="", flush=True)
    print()
    return parcial.reset_index(drop=True)


def calibrar_final(proc: str, usuario: str, X: np.ndarray, manifesto: pd.DataFrame, experimento_id: str, dataset_fp: str) -> dict:
    y = (manifesto.Usuario.to_numpy(str) == usuario).astype(int)
    ss, rr = oof_duplo_5fold(X, y)
    zs, ms, es = normalizar_scores(ss, y); zr, mr, er = normalizar_scores(rr, y); zf = ALPHA_FUSAO * zs + (1 - ALPHA_FUSAO) * zr
    ls, lr, lf = melhor_limiar_d(zs, y), melhor_limiar_d(zr, y), melhor_limiar_d(zf, y)
    svm, rf = criar_svm(), criar_rf(); svm.fit(X, y); rf.fit(X, y)
    pasta = MODELOS_DIR / proc / usuario; pasta.mkdir(parents=True, exist_ok=True)
    salvar_joblib(svm, pasta / "svm_rbf.joblib"); salvar_joblib(rf, pasta / "random_forest.joblib")
    cfg = {
        "Versao": VERSAO, "Experimento_ID": experimento_id, "Dataset_Fingerprint": dataset_fp, "Processamento": proc, "Politica": "D",
        "Feature_Config_Hash": FEATURE_CONFIG_HASHES[proc],
        "SVM": {"mediana_impostor_oof": ms, "escala_robusta_impostor_oof": es, "limiar_z": float(ls["Limiar"])},
        "RF": {"mediana_impostor_oof": mr, "escala_robusta_impostor_oof": er, "limiar_z": float(lr["Limiar"])},
        "FUSAO": {"alpha": ALPHA_FUSAO, "peso_svm": ALPHA_FUSAO, "peso_rf": 1 - ALPHA_FUSAO, "limiar_z": float(lf["Limiar"])},
    }
    salvar_json(cfg, pasta / "configuracao.json")
    return {"Processamento": proc, "Usuario": usuario, "Limiar_SVM_Z": ls["Limiar"], "Limiar_RF_Z": lr["Limiar"], "Limiar_Fusao_Z": lf["Limiar"], "GA_SVM_Calibracao": ls["GA"], "IA_SVM_Calibracao": ls["IA"], "GA_RF_Calibracao": lr["GA"], "IA_RF_Calibracao": lr["IA"], "GA_Fusao_Calibracao": lf["GA"], "IA_Fusao_Calibracao": lf["IA"]}


def versoes_ambiente() -> dict:
    return {"python": sys.version.split()[0], "platform": platform.platform(), "numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__, "pywavelets": pywt.__version__, "soundfile": sf.__version__, "joblib": joblib.__version__}


def main() -> None:
    ini = time.perf_counter()
    manifesto = construir_manifesto_treino(); dataset_fp = fingerprint_ordem(manifesto); content_fp = fingerprint_conteudo(manifesto)
    experimento_id = hash_obj({"versao": VERSAO, "dataset": dataset_fp, "features": FEATURE_CONFIG_HASHES, "modelos": MODEL_CONFIG, "validacao": VALIDATION_CONFIG})
    salvar_json({"Versao": VERSAO, "Estado": "EM_EXECUCAO", "Processamentos": list(PROCESSAMENTOS), "Politica": "D", "Amostras_Treino": N_TREINO_TOTAL, "Feature_Config": FEATURE_CONFIGS, "Feature_Config_Hashes": FEATURE_CONFIG_HASHES, "Model_Config": MODEL_CONFIG, "Validation_Config": VALIDATION_CONFIG, "Dataset_Fingerprint": dataset_fp, "Content_Fingerprint": content_fp, "Experimento_ID": experimento_id, "Ambiente": versoes_ambiente()}, SAIDA_DIR / "configuracao_global.json")
    Xs = {p: obter_features(manifesto, p) for p in PROCESSAMENTOS}
    blocos = []
    for proc in PROCESSAMENTOS:
        for usuario in USUARIOS:
            parcial, completos = carregar_checkpoint(proc, usuario, experimento_id)
            blocos.append(nested_usuario(proc, usuario, Xs[proc], manifesto, experimento_id, parcial, completos))
    nested = pd.concat(blocos, ignore_index=True)
    if len(nested) != TOTAL_LINHAS_NESTED: raise AssertionError(f"Nested: {len(nested)} != {TOTAL_LINHAS_NESTED}")
    salvar_csv(nested, RELATORIOS_DIR / "01_resultados_CV_NESTED_detalhado.csv")
    resumo = []
    for proc in PROCESSAMENTOS:
        for usuario in USUARIOS:
            resumo.append(calibrar_final(proc, usuario, Xs[proc], manifesto, experimento_id, dataset_fp))
    salvar_csv(pd.DataFrame(resumo), RELATORIOS_DIR / "02_calibracao_final_por_usuario.csv")
    tempo = time.perf_counter() - ini
    salvar_json({"Versao": VERSAO, "Estado": "TREINAMENTO_CONCLUIDO", "Processamentos": list(PROCESSAMENTOS), "Politica": "D", "Amostras_Treino": N_TREINO_TOTAL, "Feature_Config": FEATURE_CONFIGS, "Feature_Config_Hashes": FEATURE_CONFIG_HASHES, "Model_Config": MODEL_CONFIG, "Validation_Config": VALIDATION_CONFIG, "Dataset_Fingerprint": dataset_fp, "Content_Fingerprint": content_fp, "Experimento_ID": experimento_id, "Ambiente": versoes_ambiente(), "Tempo_Total_s": tempo}, SAIDA_DIR / "configuracao_global.json")
    print(f"Concluído. Saída: {SAIDA_DIR}")


if __name__ == "__main__":
    main()
