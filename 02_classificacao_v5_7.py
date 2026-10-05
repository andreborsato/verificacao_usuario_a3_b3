from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pywt
import soundfile as sf

VERSAO = "5.7"
PROCESSAMENTOS = ("C1", "C2")
USUARIOS = tuple("ABCDEFGHIJ")
N_TREINO_POR_USUARIO = 13
EXT_INICIO, EXT_FIM = 14, 26
N_EXT_POR_USUARIO = 13
N_TREINO_TOTAL = 130
N_EXT_TOTAL = 130
FS = 44100
MIN_N = 64
WAVELET = "db4"
DWT_LEVEL = 1
DWT_MODE = "symmetric"
FATOR_DWT = 0.10
MAD_CONST = 0.6744897501960817
PERCENTUAIS = np.arange(5, 100, 5, dtype=float)
COLS_FEATURES = [f"A3_{int(q):02d}" for q in PERCENTUAIS] + [f"B3_{int(q):02d}" for q in PERCENTUAIS]
ALPHA_FUSAO = 0.75
CSV_KW = dict(index=False, encoding="utf-8-sig", float_format="%.10g")


def _ancestral_db(path: Path) -> Path | None:
    path = path.resolve()
    for p in [path] + list(path.parents):
        if p.name.lower() == "db" and p.is_dir(): return p
    return None


def resolver_base_dir() -> Path:
    env = os.environ.get("BIOMETRIA_DB")
    if env:
        p = Path(env).expanduser().resolve()
        if p.is_dir(): return p
    if "__file__" in globals():
        p = _ancestral_db(Path(__file__).resolve().parent)
        if p is not None: return p
    p = _ancestral_db(Path.cwd())
    if p is not None: return p
    colab = Path("/content/drive/MyDrive/DB")
    if colab.is_dir(): return colab.resolve()
    raise FileNotFoundError("Pasta DB não localizada.")


BASE_DIR = resolver_base_dir(); AUDIO_ROOT = BASE_DIR; SAIDA_DIR = BASE_DIR / "Algoritmo_v5.7"; MODELOS_DIR = SAIDA_DIR / "modelos"; RELATORIOS_DIR = SAIDA_DIR / "relatorios"; CACHE_DIR = SAIDA_DIR / "cache"
RELATORIOS_DIR.mkdir(parents=True, exist_ok=True); CACHE_DIR.mkdir(parents=True, exist_ok=True)


def feature_config(proc: str) -> dict:
    return {"schema": 1, "processamento": proc, "usa_dwt": proc == "C2", "fs_hz": FS, "min_n": MIN_N, "remocao_media_antes_recorte": True, "recorte_energia_inicio": 0.02, "recorte_energia_fim": 0.98, "wavelet": WAVELET if proc == "C2" else None, "dwt_level": DWT_LEVEL if proc == "C2" else None, "dwt_mode": DWT_MODE if proc == "C2" else None, "fator_dwt": FATOR_DWT if proc == "C2" else None, "mad_const": MAD_CONST if proc == "C2" else None, "threshold_mode": "soft" if proc == "C2" else None, "percentuais_a3_b3": [int(v) for v in PERCENTUAIS], "n_features": 38, "politica_zeros_b3": "propagar_ultimo_sinal_nao_nulo; zeros_iniciais_recebem_primeiro_sinal_nao_nulo", "indice_b3": "idx+2_amostras"}


def hash_obj(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


FEATURE_CONFIG_HASHES = {p: hash_obj(feature_config(p)) for p in PROCESSAMENTOS}


def salvar_csv(df, path):
    tmp = path.with_suffix(path.suffix + ".tmp"); df.to_csv(tmp, **CSV_KW); os.replace(tmp, path)


def sha256(path: Path) -> str:
    h = hashlib.sha256();
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""): h.update(b)
    return h.hexdigest()


def validar_wav(path: Path):
    i = sf.info(str(path)); erros = []
    if i.format != "WAV": erros.append("formato")
    if i.subtype != "PCM_16": erros.append("PCM16")
    if i.channels != 1: erros.append("mono")
    if i.samplerate != FS: erros.append("fs")
    if i.frames < MIN_N: erros.append("curto")
    if erros: raise ValueError(f"{path.name}: {erros}")
    return i


def nomes_externos(): return [f"{u}_EXT_{n:02d}.wav" for u in USUARIOS for n in range(EXT_INICIO, EXT_FIM + 1)]


def construir_manifesto_externo() -> pd.DataFrame:
    nomes = nomes_externos(); encontrados = {}
    for p in AUDIO_ROOT.rglob("*.wav"):
        if p.name in nomes:
            if p.name in encontrados: raise RuntimeError(f"Nome duplicado: {p.name}")
            encontrados[p.name] = p
    faltam = [n for n in nomes if n not in encontrados]
    if faltam: raise FileNotFoundError(f"Faltam {len(faltam)} externos")
    rows = []
    for nome in nomes:
        p = encontrados[nome]; info = validar_wav(p); partes = p.stem.split("_")
        rows.append({"Usuario": partes[0], "Numero": int(partes[-1]), "Arquivo": nome, "Caminho": str(p.resolve()), "SHA256": sha256(p), "Frames": int(info.frames)})
    df = pd.DataFrame(rows)
    if df.SHA256.duplicated().any(): raise RuntimeError("Conteúdo duplicado no externo")
    treino = pd.read_csv(SAIDA_DIR / "manifesto_treino.csv")
    comum = set(treino.SHA256.astype(str)) & set(df.SHA256.astype(str))
    if comum: raise RuntimeError(f"Overlap treino/externo: {len(comum)}")
    salvar_csv(df, SAIDA_DIR / "manifesto_externo.csv")
    return df


def carregar_sinal(path: Path):
    validar_wav(path); x, fs = sf.read(str(path), dtype="float64", always_2d=False); x = np.asarray(x, float); x -= np.mean(x); return x


def recortar_energia(x):
    e = np.cumsum(x*x); total = e[-1]; i = int(np.searchsorted(e, .02*total, "left")); j = int(np.searchsorted(e, .98*total, "left"))+1; return x[i:min(j,len(x))]


def sigma_mad(d):
    s = np.median(np.abs(d-np.median(d))) / MAD_CONST
    if not np.isfinite(s) or s <= np.finfo(float).eps: s = np.std(d, ddof=1)
    return float(s)


def aplicar_dwt(x):
    a,d = pywt.wavedec(x,WAVELET,level=DWT_LEVEL,mode=DWT_MODE); lim = FATOR_DWT*sigma_mad(d)*np.sqrt(2*np.log(max(len(d),2))); y = pywt.waverec([a,pywt.threshold(d,lim,mode="soft")],WAVELET,mode=DWT_MODE)[:len(x)]; y=np.asarray(y,float); y-=np.mean(y); return y


def a3(x):
    e=np.cumsum(x*x); idx=np.searchsorted(e,(PERCENTUAIS/100)*e[-1],"left"); return np.minimum(idx+1,len(x))/len(x)


def b3(x):
    s=np.sign(x-np.mean(x)).astype(np.int8); nz=np.flatnonzero(s)
    if nz.size==0: raise ValueError("B3")
    s[:nz[0]]=s[nz[0]]
    for k in range(nz[0]+1,len(s)):
        if s[k]==0: s[k]=s[k-1]
    z=(s[:-1]*s[1:]<0).astype(int); total=int(z.sum())
    if total==0: raise ValueError("B3")
    c=np.cumsum(z); idx=np.searchsorted(c,(PERCENTUAIS/100)*total,"left"); return np.minimum(idx+2,len(x))/len(x)


def extrair_features(path: Path, proc: str):
    x=recortar_energia(carregar_sinal(path));
    if proc=="C2": x=aplicar_dwt(x)
    v=np.r_[a3(x),b3(x)]
    if v.shape!=(38,): raise ValueError(path.name)
    return v


def obter_features(manifesto: pd.DataFrame, proc: str):
    npz=CACHE_DIR/f"features_externo_{proc}.npz"; cfg=FEATURE_CONFIG_HASHES[proc]; shas=manifesto.SHA256.to_numpy(str)
    if npz.exists():
        try:
            d=np.load(npz,allow_pickle=False)
            if d["X"].shape==(N_EXT_TOTAL,38) and np.array_equal(d["sha"],shas) and str(d["feature_config_hash"].item())==cfg: return d["X"]
        except Exception: pass
    X=np.stack([extrair_features(Path(p),proc) for p in manifesto.Caminho]); np.savez_compressed(npz,X=X,sha=shas,feature_config_hash=np.asarray(cfg)); return X


def score_svm(modelo,X): return np.asarray(modelo.decision_function(X),float).reshape(-1)

def score_rf(modelo,X):
    pr=modelo.predict_proba(X); classes=list(modelo.named_steps["modelo"].classes_); return np.asarray(pr[:,classes.index(1)],float)


def resultado(classe, aceita):
    if classe==1: return "GA" if aceita else "GR"
    return "IA" if aceita else "IR"


def main():
    cfg=json.loads((SAIDA_DIR/"configuracao_global.json").read_text(encoding="utf-8")); exp=str(cfg["Experimento_ID"])
    if str(cfg.get("Versao"))!=VERSAO: raise RuntimeError("Versão incompatível")
    manifesto=construir_manifesto_externo(); Xs={p:obter_features(manifesto,p) for p in PROCESSAMENTOS}; reais=manifesto.Usuario.to_numpy(str); rows=[]
    for proc in PROCESSAMENTOS:
        if str(cfg["Feature_Config_Hashes"][proc]) != FEATURE_CONFIG_HASHES[proc]: raise RuntimeError(f"Hash {proc} incompatível")
        X=Xs[proc]
        for usuario in USUARIOS:
            pasta=MODELOS_DIR/proc/usuario; c=json.loads((pasta/"configuracao.json").read_text(encoding="utf-8")); svm=joblib.load(pasta/"svm_rbf.joblib"); rf=joblib.load(pasta/"random_forest.joblib")
            rs=score_svm(svm,X); rr=score_rf(rf,X); zs=(rs-float(c["SVM"]["mediana_impostor_oof"]))/float(c["SVM"]["escala_robusta_impostor_oof"]); zr=(rr-float(c["RF"]["mediana_impostor_oof"]))/float(c["RF"]["escala_robusta_impostor_oof"]); zf=ALPHA_FUSAO*zs+(1-ALPHA_FUSAO)*zr
            ls=float(c["SVM"]["limiar_z"]); lr=float(c["RF"]["limiar_z"]); lf=float(c["FUSAO"]["limiar_z"])
            for k,r in manifesto.iterrows():
                classe=int(reais[k]==usuario); as_=bool(zs[k]>=ls); ar=bool(zr[k]>=lr); af=bool(zf[k]>=lf)
                rows.append({"Experimento_ID":exp,"Processamento":proc,"Politica":"D","Modelo_Dono":usuario,"Usuario_Real":str(r.Usuario),"Arquivo":str(r.Arquivo),"Numero":int(r.Numero),"Classe_Real":classe,"Score_SVM_Raw":float(rs[k]),"Score_SVM_Z":float(zs[k]),"Limiar_SVM_Z":ls,"Decisao_SVM":"ACEITA" if as_ else "REJEITA","Resultado_SVM":resultado(classe,as_),"Score_RF_Raw":float(rr[k]),"Score_RF_Z":float(zr[k]),"Limiar_RF_Z":lr,"Decisao_RF":"ACEITA" if ar else "REJEITA","Resultado_RF":resultado(classe,ar),"Alpha_Fusao":ALPHA_FUSAO,"Peso_SVM":ALPHA_FUSAO,"Peso_RF":1-ALPHA_FUSAO,"Score_Fusao_Z":float(zf[k]),"Limiar_Fusao_Z":lf,"Decisao_Fusao":"ACEITA" if af else "REJEITA","Resultado_Fusao":resultado(classe,af)})
    df=pd.DataFrame(rows)
    esperado=len(PROCESSAMENTOS)*len(USUARIOS)*N_EXT_TOTAL
    if len(df)!=esperado: raise AssertionError(f"{len(df)} != {esperado}")
    salvar_csv(df,RELATORIOS_DIR/"03_classificacao_externa_detalhada.csv")
    print(f"Concluído. {len(df)} decisões salvas em {RELATORIOS_DIR}")


if __name__=="__main__": main()
