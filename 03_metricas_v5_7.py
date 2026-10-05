from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

VERSAO="5.7"
PROCESSAMENTOS=("C1","C2")
USUARIOS=tuple("ABCDEFGHIJ")
METODOS={"SVM_RBF":("Decisao_SVM","Score_SVM_Z"),"RF":("Decisao_RF","Score_RF_Z"),"FUSAO_SVM_RF":("Decisao_Fusao","Score_Fusao_Z")}
N_TREINO_POR_USUARIO=13
N_EXT_POR_USUARIO=13
N_TREINO_TOTAL=130
N_EXT_TOTAL=130
OUTER_POR_PROC_METODO=len(USUARIOS)*N_TREINO_TOTAL
TOTAL_INTERNO=len(PROCESSAMENTOS)*len(METODOS)*OUTER_POR_PROC_METODO
EXTERNO_POR_PROC=len(USUARIOS)*N_EXT_TOTAL
TOTAL_EXTERNO=len(PROCESSAMENTOS)*EXTERNO_POR_PROC
CSV_KW=dict(index=False,encoding="utf-8-sig",float_format="%.10g")


def _ancestral_db(path:Path):
    path=path.resolve()
    for p in [path]+list(path.parents):
        if p.name.lower()=="db" and p.is_dir(): return p


def resolver_base_dir():
    env=os.environ.get("BIOMETRIA_DB")
    if env and Path(env).is_dir(): return Path(env).resolve()
    if "__file__" in globals():
        p=_ancestral_db(Path(__file__).resolve().parent)
        if p:return p
    p=_ancestral_db(Path.cwd())
    if p:return p
    q=Path("/content/drive/MyDrive/DB")
    if q.is_dir():return q.resolve()
    raise FileNotFoundError("DB")


BASE_DIR=resolver_base_dir(); SAIDA_DIR=BASE_DIR/"Algoritmo_v5.7"; RELATORIOS_DIR=SAIDA_DIR/"relatorios"


def salvar_csv(df,path):
    tmp=path.with_suffix(path.suffix+".tmp"); df.to_csv(tmp,**CSV_KW); os.replace(tmp,path)


def eer_pct(y,s):
    fpr,tpr,_=roc_curve(y,s); fnr=1-tpr; k=int(np.argmin(np.abs(fpr-fnr))); return 100*float((fpr[k]+fnr[k])/2)


def auc_eer_macro(df,score_col):
    aa=[]; ee=[]
    for _,g in df.groupby("Modelo_Dono",sort=False):
        y=g.Classe_Real.to_numpy(int); s=g[score_col].to_numpy(float); aa.append(float(roc_auc_score(y,s))); ee.append(eer_pct(y,s))
    return float(np.mean(aa)),float(np.mean(ee))


def linha_metricas(df,metodo,usuario=None):
    dec,score=METODOS[metodo]; g=df[df.Classe_Real==1]; i=df[df.Classe_Real==0]
    ga=int((g[dec]=="ACEITA").sum()); gr=int((g[dec]=="REJEITA").sum()); ia=int((i[dec]=="ACEITA").sum()); ir=int((i[dec]=="REJEITA").sum())
    if usuario is None: auc,eer=auc_eer_macro(df,score)
    else:
        y=df.Classe_Real.to_numpy(int); s=df[score].to_numpy(float); auc=float(roc_auc_score(y,s)); eer=eer_pct(y,s)
    r={"Metodo":metodo,"Genuinas":ga+gr,"GA":ga,"GR":gr,"TAR_pct":100*ga/(ga+gr),"FRR_pct":100*gr/(ga+gr),"Impostoras":ia+ir,"IA":ia,"IR":ir,"FAR_pct":100*ia/(ia+ir),"AUC":auc,"EER_pct":eer}
    if usuario is not None:r["Usuario"]=usuario
    return r


def metricas_internas(df):
    rows=[]
    for (proc,met),g in df.groupby(["Processamento","Metodo"],sort=True):
        gg=g[g.Classe_Real==1]; ii=g[g.Classe_Real==0]; ga=int((gg.Decisao=="ACEITA").sum()); gr=int((gg.Decisao=="REJEITA").sum()); ia=int((ii.Decisao=="ACEITA").sum()); ir=int((ii.Decisao=="REJEITA").sum()); auc,eer=auc_eer_macro(g,"Score_Z")
        rows.append({"Processamento":proc,"Politica":"D","Metodo":met,"Genuinas":ga+gr,"GA":ga,"GR":gr,"TAR_pct":100*ga/(ga+gr),"FRR_pct":100*gr/(ga+gr),"Impostoras":ia+ir,"IA":ia,"IR":ir,"FAR_pct":100*ia/(ia+ir),"AUC":auc,"EER_pct":eer})
    return pd.DataFrame(rows)


def discordancias(df):
    d=df[df.Decisao_RF!=df.Decisao_SVM].copy()
    def tipo(r):
        if r.Decisao_RF=="REJEITA":return "RF_REJEITA_SVM_ACEITA_GENUINA" if r.Classe_Real==1 else "RF_REJEITA_SVM_ACEITA_IMPOSTORA"
        return "SVM_REJEITA_RF_ACEITA_GENUINA" if r.Classe_Real==1 else "SVM_REJEITA_RF_ACEITA_IMPOSTORA"
    d["Tipo_Discordancia"]=d.apply(tipo,axis=1)
    resumo=d.groupby(["Processamento","Tipo_Discordancia"]).size().reset_index(name="Quantidade")
    return d,resumo


def erros(df):
    fas=[]; frs=[]
    for proc,gp in df.groupby("Processamento",sort=True):
        for met,(dec,score) in METODOS.items():
            fa=gp[(gp.Classe_Real==0)&(gp[dec]=="ACEITA")].copy(); fa.insert(0,"Metodo",met); fas.append(fa)
            fr=gp[(gp.Classe_Real==1)&(gp[dec]=="REJEITA")].copy(); fr.insert(0,"Metodo",met); frs.append(fr)
    return pd.concat(fas,ignore_index=True),pd.concat(frs,ignore_index=True)


def complementaridade(df):
    ug=[]; ui=[]; res=[]
    for proc,gp in df.groupby("Processamento",sort=True):
        for usuario,g in gp[gp.Classe_Real==1].groupby("Modelo_Dono",sort=True):
            so=(g.Decisao_SVM=="ACEITA")&(g.Decisao_RF=="REJEITA"); ro=(g.Decisao_RF=="ACEITA")&(g.Decisao_SVM=="REJEITA"); br=(g.Decisao_RF=="REJEITA")&(g.Decisao_SVM=="REJEITA")
            ug.append({"Processamento":proc,"Usuario":usuario,"GA_SVM":int((g.Decisao_SVM=="ACEITA").sum()),"GA_RF":int((g.Decisao_RF=="ACEITA").sum()),"GA_FUSAO":int((g.Decisao_Fusao=="ACEITA").sum()),"SVM_aceita_RF_rejeita":int(so.sum()),"Fusao_aceita_dessas":int((so&(g.Decisao_Fusao=="ACEITA")).sum()),"RF_aceita_SVM_rejeita":int(ro.sum()),"Fusao_aceita_dessas_RF":int((ro&(g.Decisao_Fusao=="ACEITA")).sum()),"Ambos_rejeitam":int(br.sum()),"Fusao_recupera_ambos_rejeitam":int((br&(g.Decisao_Fusao=="ACEITA")).sum())})
        for usuario,g in gp[gp.Classe_Real==0].groupby("Modelo_Dono",sort=True):
            so=(g.Decisao_SVM=="ACEITA")&(g.Decisao_RF=="REJEITA"); ro=(g.Decisao_RF=="ACEITA")&(g.Decisao_SVM=="REJEITA")
            ui.append({"Processamento":proc,"Usuario":usuario,"IA_SVM":int((g.Decisao_SVM=="ACEITA").sum()),"IA_RF":int((g.Decisao_RF=="ACEITA").sum()),"IA_FUSAO":int((g.Decisao_Fusao=="ACEITA").sum()),"SVM_aceita_RF_rejeita":int(so.sum()),"Fusao_aceita_dessas":int((so&(g.Decisao_Fusao=="ACEITA")).sum()),"RF_aceita_SVM_rejeita":int(ro.sum()),"Fusao_aceita_dessas_RF":int((ro&(g.Decisao_Fusao=="ACEITA")).sum())})
        for classe,nome in [(1,"GENUINA"),(0,"IMPOSTORA")]:
            d=gp[gp.Classe_Real==classe]
            casos=[("SVM_ACEITA_RF_REJEITA",(d.Decisao_SVM=="ACEITA")&(d.Decisao_RF=="REJEITA")),("RF_ACEITA_SVM_REJEITA",(d.Decisao_RF=="ACEITA")&(d.Decisao_SVM=="REJEITA")),("AMBOS_ACEITAM",(d.Decisao_RF=="ACEITA")&(d.Decisao_SVM=="ACEITA")),("AMBOS_REJEITAM",(d.Decisao_RF=="REJEITA")&(d.Decisao_SVM=="REJEITA"))]
            for caso,m in casos:
                sub=d[m]; res.append({"Processamento":proc,"Classe":nome,"Caso":caso,"Quantidade":len(sub),"Fusao_Aceita":int((sub.Decisao_Fusao=="ACEITA").sum()),"Fusao_Rejeita":int((sub.Decisao_Fusao=="REJEITA").sum())})
    return pd.DataFrame(ug),pd.DataFrame(ui),pd.DataFrame(res)


def delta_c2_c1(tabela):
    rows=[]
    for met in METODOS:
        c1=tabela[(tabela.Processamento=="C1")&(tabela.Metodo==met)].iloc[0]; c2=tabela[(tabela.Processamento=="C2")&(tabela.Metodo==met)].iloc[0]
        rows.append({"Metodo":met,"GA_C1":c1.GA,"GA_C2":c2.GA,"Delta_GA_C2_menos_C1":c2.GA-c1.GA,"IA_C1":c1.IA,"IA_C2":c2.IA,"Delta_IA_C2_menos_C1":c2.IA-c1.IA,"TAR_C1":c1.TAR_pct,"TAR_C2":c2.TAR_pct,"Delta_TAR_C2_menos_C1":c2.TAR_pct-c1.TAR_pct,"FAR_C1":c1.FAR_pct,"FAR_C2":c2.FAR_pct,"Delta_FAR_C2_menos_C1":c2.FAR_pct-c1.FAR_pct,"AUC_C1":c1.AUC,"AUC_C2":c2.AUC,"Delta_AUC_C2_menos_C1":c2.AUC-c1.AUC,"EER_C1":c1.EER_pct,"EER_C2":c2.EER_pct,"Delta_EER_C2_menos_C1":c2.EER_pct-c1.EER_pct})
    return pd.DataFrame(rows)


def main():
    cfg=json.loads((SAIDA_DIR/"configuracao_global.json").read_text(encoding="utf-8")); exp=str(cfg["Experimento_ID"])
    interno=pd.read_csv(RELATORIOS_DIR/"01_resultados_CV_NESTED_detalhado.csv"); externo=pd.read_csv(RELATORIOS_DIR/"03_classificacao_externa_detalhada.csv")
    if len(interno)!=TOTAL_INTERNO: raise AssertionError(f"Interno {len(interno)} != {TOTAL_INTERNO}")
    if len(externo)!=TOTAL_EXTERNO: raise AssertionError(f"Externo {len(externo)} != {TOTAL_EXTERNO}")
    if set(interno.Experimento_ID.astype(str))!={exp} or set(externo.Experimento_ID.astype(str))!={exp}: raise AssertionError("Experimento_ID")
    tin=metricas_internas(interno)
    ext=[]; por=[]
    for proc,gp in externo.groupby("Processamento",sort=True):
        for met in METODOS:
            r=linha_metricas(gp,met); r.update({"Processamento":proc,"Politica":"D"}); ext.append(r)
            for usuario,gu in gp.groupby("Modelo_Dono",sort=True):
                q=linha_metricas(gu,met,usuario); q.update({"Processamento":proc,"Politica":"D"}); por.append(q)
    tex=pd.DataFrame(ext); tpu=pd.DataFrame(por)
    fa,fr=erros(externo); disc,resdisc=discordancias(externo); cg,ci,cr=complementaridade(externo); delta=delta_c2_c1(tex)
    salvar_csv(tin,RELATORIOS_DIR/"04_metricas_INTERNAS_NESTED.csv"); salvar_csv(tex,RELATORIOS_DIR/"05_metricas_EXTERNAS_globais.csv"); salvar_csv(tpu,RELATORIOS_DIR/"06_metricas_EXTERNAS_por_usuario.csv"); salvar_csv(fa,RELATORIOS_DIR/"07_falsas_aceitacoes_EXTERNAS.csv"); salvar_csv(fr,RELATORIOS_DIR/"08_falsas_rejeicoes_EXTERNAS.csv"); salvar_csv(disc,RELATORIOS_DIR/"09_discordancias_RF_vs_SVM_detalhadas.csv"); salvar_csv(resdisc,RELATORIOS_DIR/"10_discordancias_RF_vs_SVM_resumo.csv"); salvar_csv(cg,RELATORIOS_DIR/"11_complementaridade_genuinas_por_usuario.csv"); salvar_csv(ci,RELATORIOS_DIR/"12_complementaridade_impostoras_por_usuario.csv"); salvar_csv(cr,RELATORIOS_DIR/"13_resumo_complementaridade.csv"); salvar_csv(tex[["Processamento","Politica","Metodo","GA","GR","TAR_pct","FRR_pct","IA","IR","FAR_pct","AUC","EER_pct"]],RELATORIOS_DIR/"14_comparacao_C1_C2_D_SVM_RF_FUSAO.csv"); salvar_csv(delta,RELATORIOS_DIR/"15_delta_C2_menos_C1.csv")
    print("\nRESULTADOS EXTERNOS — C1-D x C2-D")
    print(tex[["Processamento","Metodo","GA","GR","IA","IR","TAR_pct","FRR_pct","FAR_pct","AUC","EER_pct"]].to_string(index=False))
    print(f"\nSaída: {RELATORIOS_DIR}")


if __name__=="__main__": main()
