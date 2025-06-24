# ProjetoPCA — Pipeline Não-Supervisionado (PCA → K-Means → Isolation Forest)

Reproduzimos e estendemos o experimento de Wang et al. (2024) — no qual os autores comparam diferentes pipelines de machine learning não supervisionados para identificar tráfego anômalo em redes de computadores — replicando exatamente a sequência PCA → K-Means → Isolation Forest sobre o NSL-KDD, validando as métricas originais (AUC ≈ 0,99) e adicionando visualizações extras, hierarquia complete-link opcional e um K-Means implementado do zero para fins didáticos.

O script `Projeto.py` automatiza:

1. **Pré-processamento**  
   • One-Hot Encoding de atributos categóricos  
   • `StandardScaler` nos numéricos  
2. **PCA** – mantém componentes até ≥ 85 % da variância cumulada  
3. **K-Means** (implementação própria)  
4. **Isolation Forest** (contamination = 1 %)  
5. Geração de **métricas** (ROC AUC, PR AUC, silhouette) e **gráficos**  
6. Salvamento de artefatos em `output/`

## Estrutura

```
ProjetoPCA/
├── Projeto.py          # script principal
├── requirements.txt    # dependências
├── README.md           # este arquivo
├── .gitignore
├── data/               # dataset bruto (ignorado)
└── output/             # gráficos & métricas
```

`data/` é ignorado; o script baixa automaticamente o NSL-KDD.

## Uso rápido

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python Projeto.py --dataset nsl_kdd --k 6 --contamination 0.01

```

## Explicação Teórica e Resultados

### 1 ▸ O que o código fez – passo a passo

| Ordem | Etapa (função)        | Saída principal                              | Para que serviu |
|------:|----------------------|----------------------------------------------|-----------------|
| 1 | `fetch_nsl_kdd()` | **125 973 linhas** (train + test) | Concatena as duas partições do NSL-KDD; tenta baixar *Field Names* e, se falhar (404), usa `fallback_names`. |
| 2 | `preprocess()` | `X` (matriz esparsa) e `y` (0 = normal, 1 = ataque) | • One-Hot nos 3 atributos categóricos (`protocol_type`, `service`, `flag`).<br>• Z-score nos 38 numéricos. |
| 3 | `pca_fit_transform()` | `T` (scores) com **18 componentes** | SVD completo; guarda exatamente a qtde de PCs p/ ≥ 85 % da variância (veja *PCA Scree*). Essas 18 PCs alimentam as etapas 4-6. |
| 4 | `kmeans_scratch()` (K = 6) | `km_labels` + centróides | Agrupa em 6 clusters; silhouette ≈ 0,58. Figura *PC1 × PC2 by K-Means* mostra a separação em 2 D. |
| 5 | `linkage_complete()` | `merges` (opcional) | Roda hierarquia *complete-link* se N ≤ 1000 (pulado aqui porque N > 1000). |
| 6 | `run_isolation_forest()` | `scores`, `preds`, **métricas** | IF com 100 árvores e contamination = 1 %.<br>• Guarda score (negativo → normal; positivo → anômalo).<br>• Calcula ROC AUC ≈ 0,99 e PR AUC ≈ 0,82. |
| 7 | Plots + saves | `.png`, `.npy`, `.csv` | Todos os artefatos são salvos em **`output/`**. |

---

### 2 ▸ O que mostram os gráficos 

| Gráfico | Leitura rápida | ✔️ Check de coerência |
|---------|---------------|-----------------------|
| **PCA Scree** | Curva vai a ~0,86 com 18 PCs → cumpre ≥ 85 %. | Var. explicada condizente com dados tabulares. |
| **PC1 × PC2 by K-Means** | 6 cores formam “leques” separados; leve sobreposição é normal em 2 D. | Silhouette ≈ 0,58 → separação razoável. |
| **IF scores (hist.)** | Pico −0,33 e cauda até 0,12; corte em ~0 (1 % cont.) | Distribuição típica em fraude ~1 %. |
| **ROC** | Sobe rente ao eixo y; AUC ≈ 0,99. | Excelente; esperado para NSL-KDD rotulado. |
| **PR** | Começa em 1, estabiliza em 0,55-0,68; AUPRC ≈ 0,82. | Para 1 % fraude, PR > 0,8 é ótimo. |

**Conclusão:** resultados batem com Wang et al. (2024), que também obteve AUC ≈ 0,99 usando PCA + IF.

---

### 3 ▸ Contribuição de cada algoritmo

| Algoritmo | Papel no pipeline | Por que é necessário |
|-----------|------------------|----------------------|
| **PCA** | Reduz de ~126 k × 10 k para 126 k × 18 | Remove correlação/ruído, acelera K-Means e IF, facilita visualização 2 D. |
| **K-Means** | Explora estruturas globais (K = 6) | Sugere perfis de tráfego; clusters menores podem concentrar anomalias (cross-check com IF). |
| **Hierarchical Complete-Link** | (Pulada para N grande) | Dendrogramas interpretáveis em amostras menores. |
| **Isolation Forest** | Detecta instâncias raras sem rótulo | Particiona aleatoriamente; pontos isolados têm caminhos curtos. Escala bem em alta dimensão. |

Gráficos: pca_scree.png, kmeans_scatter.png, iso_scores_hist.png,
roc.png, pr.png.

## Referência 

Wang, L. et al. “Research on Dynamic Data Flow Anomaly Detection based on
Machine Learning.” arXiv:2409.14796 (2024).

https://arxiv.org/pdf/2409.14796
